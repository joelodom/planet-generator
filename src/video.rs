//! Video encoding for the headless `--video` tour — a thin wrapper around an
//! `ffmpeg` child process. We pipe raw RGBA frames to ffmpeg's stdin and it muxes
//! an H.264 MP4. This module knows nothing about wgpu/gfx: it takes plain frame
//! bytes plus dimensions/fps, keeping the CPU/GPU seam clean (the bytes come from
//! [`crate::gfx::Renderer::render_to_rgba`]).
//!
//! ffmpeg is an OPTIONAL, runtime-only dependency used solely by `--video`; the
//! normal app never touches it. If ffmpeg isn't on `PATH`, [`VideoEncoder::start`]
//! returns a clear, actionable error so `--video` exits politely rather than
//! depending on ffmpeg for normal operation (degrade, don't panic).

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

use anyhow::{Context, anyhow};

/// libx264 quality (Constant Rate Factor): lower = better/larger. 18 is visually
/// near-lossless — a good master for a clip that gets re-encoded on upload.
const VIDEO_CRF: &str = "18";
/// x264 speed/efficiency preset; `medium` balances encode time and size for an
/// offline render.
const VIDEO_PRESET: &str = "medium";
/// AAC bitrate for the muxed background-music track — transparent for stereo music
/// and well within YouTube's recommended audio range.
const AUDIO_BITRATE: &str = "192k";

/// Well-known absolute install locations to try when a bare `ffmpeg` isn't on
/// `PATH`. This matters on macOS because a Finder-launched `.app` inherits a
/// minimal `PATH` (`/usr/bin:/bin:/usr/sbin:/sbin`) that excludes Homebrew's bin
/// dir — so `ffmpeg` resolves from a terminal but not from the bundle. Override
/// with `$PLANET_FFMPEG` (full path to the binary) on any platform.
#[cfg(target_os = "macos")]
const FFMPEG_FALLBACK_PATHS: &[&str] = &[
    "/opt/homebrew/bin/ffmpeg", // Apple Silicon Homebrew
    "/usr/local/bin/ffmpeg",    // Intel Homebrew
    "/opt/local/bin/ffmpeg",    // MacPorts
];
#[cfg(not(target_os = "macos"))]
const FFMPEG_FALLBACK_PATHS: &[&str] = &[];

/// A handle to a running ffmpeg encode. Feed it [`VideoEncoder::write_frame`] per
/// rendered frame, then [`VideoEncoder::finish`] to finalize the file (a `Drop`
/// backstop finalizes too, so an error path still leaves a playable clip).
pub struct VideoEncoder {
    child: Child,
    stdin: Option<ChildStdin>,
    /// The resolved ffmpeg program (from the live pass), reused for the mux pass.
    program: OsString,
    /// Where the live pass writes. With music this is a temp file in the audio dir
    /// (muxed into `final_path` at finish); without music it *is* `final_path`.
    video_path: PathBuf,
    /// Where the finished file the user keeps lands.
    final_path: PathBuf,
    /// Staged soundtrack to mux in at finish; `None` when recording silent. Owns the
    /// temp dir (which also holds `video_path`), removed on finish/drop.
    audio: Option<StagedAudio>,
}

impl VideoEncoder {
    /// Spawn ffmpeg to read raw RGBA frames of `width`×`height` at `fps` from stdin
    /// and write an H.264 MP4 to `path`. Errors clearly if ffmpeg isn't installed.
    ///
    /// `soundtrack` is an ordered list of mp3 byte blobs (see
    /// [`crate::audio::shuffled_soundtrack`]); when non-empty it's muxed in as a
    /// looped background-music track by a fast **second pass at [`finish`]**, not
    /// against the live frame pipe — a slow render (e.g. 1080p) trickles frames in,
    /// and a looped audio input muxed live would race ahead and overflow ffmpeg's
    /// interleave buffer (ENOSPC). Music is best-effort: if the tracks can't be
    /// staged, or the mux pass fails, we keep the silent video rather than failing.
    ///
    /// [`finish`]: VideoEncoder::finish
    pub fn start(
        path: &Path,
        width: u32,
        height: u32,
        fps: u32,
        soundtrack: &[&[u8]],
    ) -> anyhow::Result<Self> {
        let size = format!("{width}x{height}");
        let fps_s = fps.to_string();
        let audio = stage_soundtrack(soundtrack);

        // The live pass encodes VIDEO ONLY. With music, write to a temp file in the
        // audio dir and mux at finish; without, write the final file directly (and
        // only then is `+faststart` worth the rewrite — the intermediate is discarded).
        let video_path = match &audio {
            Some(a) => a.dir.join("video.mp4"),
            None => path.to_path_buf(),
        };
        let args = video_args(&size, &fps_s, audio.is_none());

        let (mut child, program) = match spawn_ffmpeg(&args, &video_path) {
            Ok(ok) => ok,
            Err(e) => {
                if let Some(a) = &audio {
                    let _ = std::fs::remove_dir_all(&a.dir);
                }
                return Err(e);
            }
        };
        let stdin = child.stdin.take().context("ffmpeg stdin unavailable")?;
        Ok(Self {
            child,
            stdin: Some(stdin),
            program,
            video_path,
            final_path: path.to_path_buf(),
            audio,
        })
    }

    /// Pipe one RGBA frame (`width * height * 4` bytes) to ffmpeg.
    pub fn write_frame(&mut self, rgba: &[u8]) -> anyhow::Result<()> {
        let stdin = self.stdin.as_mut().context("encoder already finalized")?;
        stdin
            .write_all(rgba)
            .context("writing a frame to ffmpeg failed (did it exit early?)")?;
        Ok(())
    }

    /// Close ffmpeg's stdin, let the video finalize, then (if a soundtrack was
    /// staged) mux the music in. Returns the path to the finished file.
    pub fn finish(mut self) -> anyhow::Result<PathBuf> {
        // Dropping stdin closes the pipe → ffmpeg reads EOF and writes the trailer.
        self.stdin.take();
        let status = self.child.wait().context("waiting for ffmpeg to finish")?;
        if !status.success() {
            self.cleanup_audio();
            return Err(anyhow!("ffmpeg exited unsuccessfully ({status})"));
        }
        if let Some(audio) = self.audio.take() {
            if let Err(e) = self.mux_audio(&audio) {
                // Best-effort music: keep the silent video so a recording is never lost.
                tracing::warn!(error = %format!("{e:#}"), "video: muxing music failed; saving silent video");
                let saved = std::fs::rename(&self.video_path, &self.final_path)
                    .or_else(|_| std::fs::copy(&self.video_path, &self.final_path).map(|_| ()));
                let _ = std::fs::remove_dir_all(&audio.dir);
                saved.context("saving the silent fallback video")?;
                return Ok(self.final_path.clone());
            }
            let _ = std::fs::remove_dir_all(&audio.dir);
        }
        Ok(self.final_path.clone())
    }

    /// Second pass: copy the silent video and add the looped soundtrack, trimmed to
    /// the video's length. Both inputs are files here, so the audio can't outrun the
    /// video and the interleave buffer stays bounded (the live pipe couldn't).
    fn mux_audio(&self, audio: &StagedAudio) -> anyhow::Result<()> {
        let mut cmd = Command::new(&self.program);
        cmd.args(["-hide_banner", "-loglevel", "error", "-y"])
            .arg("-i").arg(&self.video_path) // input 0: the silent video
            // input 1: the playlist, looped forever; `-shortest` trims it to the video.
            .args(["-f", "concat", "-safe", "0", "-stream_loop", "-1"])
            .arg("-i").arg(&audio.list)
            .args([
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", // no re-encode of the video — fast
                "-c:a", "aac", "-b:a", AUDIO_BITRATE,
                "-shortest",
                "-movflags", "+faststart",
            ])
            .arg(&self.final_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        // Own process group so a stray Ctrl+C during finalize doesn't kill the mux.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let status = cmd.status().context("running the audio mux pass")?;
        if !status.success() {
            return Err(anyhow!("ffmpeg audio mux failed ({status})"));
        }
        Ok(())
    }

    /// Remove the staged soundtrack temp dir, if any. Idempotent (takes the value).
    fn cleanup_audio(&mut self) {
        if let Some(a) = self.audio.take() {
            let _ = std::fs::remove_dir_all(a.dir);
        }
    }
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        // Backstop for error paths that skip finish(): close stdin so ffmpeg finalizes
        // (rather than orphaning) and reap it. Harmless no-op after a real finish().
        self.stdin.take();
        let _ = self.child.wait();
        self.cleanup_audio();
    }
}

/// Build the video-only ffmpeg argument list (everything before the output path):
/// raw RGBA frames from stdin → H.264/yuv420p. `faststart` moves the moov atom up
/// front — worth it only when this output is the file the user keeps.
fn video_args(size: &str, fps_s: &str, faststart: bool) -> Vec<String> {
    let mut args: Vec<String> = [
        "-hide_banner",
        "-loglevel", "error",
        "-y", // overwrite an existing file
        // Input: raw RGBA frames from stdin at a fixed rate.
        "-f", "rawvideo",
        "-pixel_format", "rgba",
        "-video_size", size,
        "-framerate", fps_s,
        "-i", "-",
        "-an", // video pass carries no audio; music is muxed in later
        // H.264 in yuv420p for the widest player/YouTube compatibility.
        "-c:v", "libx264",
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-pix_fmt", "yuv420p",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    if faststart {
        args.extend(["-movflags", "+faststart"].map(String::from));
    }
    args
}

/// Spawn ffmpeg with `args` then `output`, video frames piped to its stdin. Resolves
/// the program from `$PLANET_FFMPEG`, then a bare `ffmpeg` on `PATH`, then the
/// well-known install locations (so a Finder-launched macOS `.app` finds a Homebrew
/// ffmpeg despite its minimal `PATH` — see FFMPEG_FALLBACK_PATHS). Returns the child
/// and the resolved program path (reused for the mux pass). ffmpeg runs in its own
/// process group so a terminal Ctrl+C doesn't kill it mid-write: WE catch Ctrl+C,
/// stop the loop, and close stdin so ffmpeg sees EOF and finalizes cleanly.
fn spawn_ffmpeg(args: &[String], output: &Path) -> anyhow::Result<(Child, OsString)> {
    let mut candidates: Vec<OsString> = Vec::new();
    if let Some(p) = std::env::var_os("PLANET_FFMPEG") {
        candidates.push(p);
    }
    candidates.push("ffmpeg".into());
    candidates.extend(FFMPEG_FALLBACK_PATHS.iter().map(Into::into));

    for program in candidates {
        let mut cmd = Command::new(&program);
        cmd.args(args)
            .arg(output)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit()); // surface ffmpeg's own errors on our stderr
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        match cmd.spawn() {
            Ok(child) => return Ok((child, program)),
            // Not at this location — try the next candidate.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            // Found ffmpeg but it failed to launch for another reason — report it.
            Err(e) => return Err(anyhow::Error::new(e).context("failed to launch ffmpeg")),
        }
    }

    Err(anyhow!(
        "ffmpeg not found on PATH or common install locations — install it to use --video \
         (macOS: `brew install ffmpeg`, Windows: https://ffmpeg.org/download.html). \
         If it's installed elsewhere, set $PLANET_FFMPEG to its full path."
    ))
}

/// A soundtrack staged to a temp dir: the directory and the ffmpeg concat list that
/// references the per-track mp3 files inside it.
struct StagedAudio {
    dir: PathBuf,
    list: PathBuf,
}

/// Write the soundtrack tracks to a temp dir and build an ffmpeg concat-demuxer
/// playlist referencing them, so ffmpeg can read the embedded mp3s (which live in
/// the binary) as a file input. Returns `None` for an empty soundtrack or on any
/// I/O error — music is best-effort, so we degrade to a silent recording (warn,
/// don't fail), mirroring the live player's "no device → run silent" behavior.
fn stage_soundtrack(tracks: &[&[u8]]) -> Option<StagedAudio> {
    if tracks.is_empty() {
        return None;
    }
    // One dir per process; the encoder removes it on finish/drop.
    let dir = std::env::temp_dir().join(format!("planet-explorer-audio-{}", std::process::id()));
    let bail = |dir: &Path, e: std::io::Error| {
        tracing::warn!(error = %e, "video: couldn't stage soundtrack; recording silent");
        let _ = std::fs::remove_dir_all(dir);
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        bail(&dir, e);
        return None;
    }
    let mut list = String::new();
    for (i, bytes) in tracks.iter().enumerate() {
        let track = dir.join(format!("track{i}.mp3"));
        if let Err(e) = std::fs::write(&track, bytes) {
            bail(&dir, e);
            return None;
        }
        // concat-demuxer line; temp paths contain no single quotes to escape.
        list.push_str(&format!("file '{}'\n", track.display()));
    }
    let list_path = dir.join("playlist.txt");
    if let Err(e) = std::fs::write(&list_path, list) {
        bail(&dir, e);
        return None;
    }
    Some(StagedAudio { dir, list: list_path })
}
