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
    path: PathBuf,
}

impl VideoEncoder {
    /// Spawn ffmpeg to read raw RGBA frames of `width`×`height` at `fps` from stdin
    /// and write an H.264 MP4 to `path`. Errors clearly if ffmpeg isn't installed.
    pub fn start(path: &Path, width: u32, height: u32, fps: u32) -> anyhow::Result<Self> {
        let size = format!("{width}x{height}");
        let fps_s = fps.to_string();

        // Try ffmpeg programs in priority order: an explicit `$PLANET_FFMPEG`
        // override, then a bare `ffmpeg` (resolved on `PATH`, respecting the user's
        // chosen build), then the well-known install locations. The fallbacks are
        // what let a Finder-launched macOS `.app` find a Homebrew ffmpeg despite its
        // minimal `PATH` (see FFMPEG_FALLBACK_PATHS).
        let mut candidates: Vec<std::ffi::OsString> = Vec::new();
        if let Some(p) = std::env::var_os("PLANET_FFMPEG") {
            candidates.push(p);
        }
        candidates.push("ffmpeg".into());
        candidates.extend(FFMPEG_FALLBACK_PATHS.iter().map(Into::into));

        for program in &candidates {
            let mut cmd = Command::new(program);
            cmd.args([
                "-hide_banner",
                "-loglevel", "error",
                "-y", // overwrite an existing file
                // Input: raw RGBA frames from stdin at a fixed rate.
                "-f", "rawvideo",
                "-pixel_format", "rgba",
                "-video_size", &size,
                "-framerate", &fps_s,
                "-i", "-",
                "-an", // no audio track
                // Output: H.264 in yuv420p for the widest player/YouTube compatibility,
                // with the moov atom up front so it previews/streams before fully written.
                "-c:v", "libx264",
                "-preset", VIDEO_PRESET,
                "-crf", VIDEO_CRF,
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ])
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit()); // surface ffmpeg's own errors on our stderr
            // Put ffmpeg in its own process group so a terminal Ctrl+C doesn't kill it
            // mid-write: WE catch Ctrl+C, stop the loop, and close stdin so ffmpeg sees
            // EOF and finalizes the file cleanly. (Windows handles its own console
            // signals; the `ctrlc` handler still fires for our process there.)
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                cmd.process_group(0);
            }

            match cmd.spawn() {
                Ok(mut child) => {
                    let stdin = child.stdin.take().context("ffmpeg stdin unavailable")?;
                    return Ok(Self { child, stdin: Some(stdin), path: path.to_path_buf() });
                }
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

    /// Pipe one RGBA frame (`width * height * 4` bytes) to ffmpeg.
    pub fn write_frame(&mut self, rgba: &[u8]) -> anyhow::Result<()> {
        let stdin = self.stdin.as_mut().context("encoder already finalized")?;
        stdin
            .write_all(rgba)
            .context("writing a frame to ffmpeg failed (did it exit early?)")?;
        Ok(())
    }

    /// Close ffmpeg's stdin and wait for it to finalize the MP4. Returns the path.
    pub fn finish(mut self) -> anyhow::Result<PathBuf> {
        // Dropping stdin closes the pipe → ffmpeg reads EOF and writes the trailer.
        self.stdin.take();
        let status = self.child.wait().context("waiting for ffmpeg to finish")?;
        if !status.success() {
            return Err(anyhow!("ffmpeg exited unsuccessfully ({status})"));
        }
        Ok(self.path.clone())
    }
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        // Backstop for error paths that skip finish(): close stdin so ffmpeg finalizes
        // (rather than orphaning) and reap it. Harmless no-op after a real finish().
        self.stdin.take();
        let _ = self.child.wait();
    }
}
