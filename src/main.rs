//! planet-explorer — a procedural planet you can fly from orbit down to the
//! grass, generated entirely from a seed.
//!
//! ```text
//! cargo run                 # random seed
//! cargo run -- --seed 12345 # reproducible planet
//! ```
//!
//! Architecture (phase one of a longer roadmap):
//!   planet   — seeded source of truth: terrain, biomes, sun, atmosphere
//!   mesh     — turns planet samples into triangles + vegetation instances
//!   lod      — cube-sphere quadtree + background meshing pool
//!   camera   — Google-Earth-style focus-orbit navigation (pan/zoom/rotate/tilt)
//!   gfx      — wgpu renderer (sky / terrain / vegetation / water)
//!
//! Each system queries `Planet` for ground truth without touching the others,
//! which is what makes adding animals, NPCs, weather, etc. later tractable.

mod audio;
mod camera;
mod flora;
mod font8x8;
mod gfx;
mod lod;
mod logging;
mod mesh;
mod models;
mod overlay;
mod planet;
mod settings;
mod tour;
mod units;
mod video;
#[cfg(test)]
mod tests;

use camera::{Camera, KeyAction};
use gfx::{Globals, Renderer};
use glam::Vec3;
use lod::Streamer;
use planet::Planet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use anyhow::Context;
use tracing::{debug, info, warn};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

const SUN_AMBIENT: f32 = 0.32;
/// Cap chunk requests per frame so a fast camera can't flood the work queue.
/// (The resident-chunk cap is a runtime graphics setting — see `settings.rs`.)
const MAX_REQUESTS_PER_FRAME: usize = 64;

/// Soundtrack playback volume (0..1).
const AUDIO_VOLUME: f32 = 0.5;
/// Initial window size, logical pixels.
const WINDOW_WIDTH: f64 = 1280.0;
const WINDOW_HEIGHT: f64 = 800.0;
/// Idle time after launch before the guided tour auto-starts (attract mode) — only
/// if the user hasn't pressed anything yet.
const AUTO_TOUR_IDLE_SECONDS: f32 = 5.0;
/// Aggregate a performance sample to the log this often.
const PERF_SAMPLE_SECONDS: f32 = 2.0;
/// Refresh the window-title HUD this often.
const TITLE_UPDATE_SECONDS: f32 = 0.4;
/// A frame slower than this is logged as a hitch ...
const FRAME_HITCH_MS: f32 = 120.0;
/// ... but at most once per this interval, to avoid log spam.
const HITCH_LOG_COOLDOWN: f32 = 1.0;
/// Bytes per MiB, for the resident-memory readout (HUD + perf log).
const BYTES_PER_MIB: f32 = 1024.0 * 1024.0;

// --- Headless `--video` recorder (see `parse_video` / `run_video`) ---
/// Default recording resolution — 1080p, YouTube's 16:9 standard.
const VIDEO_WIDTH: u32 = 1920;
const VIDEO_HEIGHT: u32 = 1080;
/// Default recording framerate.
const VIDEO_FPS: u32 = 60;
/// Default detail preset for recording. Video isn't real-time bound, so we record
/// at high detail regardless of host GPU; "High" keeps geometry crisp with an 8 GB
/// cache budget that stays well clear of swap on common (16/18 GB) hosts. Override
/// with `--video-preset <name>` (e.g. `--video-preset "Very High"`).
const VIDEO_PRESET: &str = "High";
/// Per recorded frame, let the chunk streamer load the visible terrain/vegetation
/// before capture — bounded so a chunk that never meshes can't hang a frame.
const VIDEO_SETTLE_MAX_MS: u128 = 2000;
const VIDEO_SETTLE_SLEEP_MS: u64 = 15;
/// Heartbeat progress line every this many recorded seconds (the tour also logs a
/// line at each phase change).
const VIDEO_LOG_EVERY_SECONDS: u64 = 5;

// --- Tour finale: once every biome has been toured, pull back to a full-globe view
// from space, rotate slowly, then fade video + audio out and finalize. ---
/// Eye distance above the surface for the finale space view (× PLANET_RADIUS) — the
/// full-globe framing, matching the camera's opening view (START_DISTANCE_FACTOR).
const FINALE_SPACE_DISTANCE_FACTOR: f32 = 2.2;
/// Ease the camera from the low cruise out to the space view over this long.
const FINALE_PULLBACK_SECONDS: f32 = 4.0;
/// Slow spin in space at full brightness (includes the pull-back ease at its start).
const FINALE_ROTATE_SECONDS: f32 = 10.0;
/// Then fade the video to black and the audio to silence over this long, still
/// rotating. The audio fade is applied by the encoder's mux pass over this window.
const FINALE_FADE_SECONDS: f32 = 10.0;
/// Planet spin rate during the finale (rad/s about the pole) — slow and calm.
const FINALE_ROTATE_RATE: f32 = 0.06;

fn main() -> anyhow::Result<()> {
    if std::env::args().skip(1).any(|a| a == "--version" || a == "-V") {
        println!(
            "planet-explorer {} ({}) built {}",
            env!("CARGO_PKG_VERSION"),
            env!("GIT_HASH"),
            env!("BUILD_DATE")
        );
        return Ok(());
    }

    let log_path = logging::init();
    let seed = parse_seed();
    let unit_system = parse_units();
    let video = parse_video();
    let planet = Arc::new(Planet::new(seed));

    // Console banner (handy when launched from a terminal; invisible under
    // Finder — the log file is the durable record).
    println!("planet-explorer {} ({})", env!("CARGO_PKG_VERSION"), env!("GIT_HASH"));
    println!("  seed       : {seed}");
    let (sx, sy, sz) = (planet.sun_dir.x, planet.sun_dir.y, planet.sun_dir.z);
    println!("  sun        : ({sx:.2}, {sy:.2}, {sz:.2})");
    println!("  reproduce  : cargo run -- --seed {seed}");
    println!("  units      : {} (use --units us for imperial)", unit_system.label());
    println!("  log        : {}", log_path.display());
    if video.is_none() {
        print_controls();
    }

    info!(
        seed,
        sun = ?planet.sun_dir,
        version = env!("CARGO_PKG_VERSION"),
        commit = env!("GIT_HASH"),
        built = env!("BUILD_DATE"),
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        log = %log_path.display(),
        "planet-explorer starting"
    );

    // Headless video mode: render the guided tour to an MP4 and exit — no window,
    // no event loop. (`--video`, optionally `--video-out/-size/-fps/-seconds`.)
    if let Some(opts) = video {
        return run_video(seed, planet, unit_system, opts);
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(seed, planet, unit_system);
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Parse `--units <metric|us>` (also `--imperial`); defaults to metric.
fn parse_units() -> units::Units {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--imperial" {
            return units::Units::Us;
        }
        if a == "--units" && let Some(v) = args.next() {
            if let Some(u) = units::Units::parse(&v) {
                return u;
            }
            eprintln!("warning: unknown --units '{v}', using metric");
        }
    }
    units::Units::Metric
}

fn parse_seed() -> u64 {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if (a == "--seed" || a == "-s") && let Some(v) = args.next() {
            if let Ok(n) = v.parse::<u64>() {
                return n;
            }
            eprintln!("warning: could not parse seed '{v}', using random");
        }
    }
    rand::random::<u64>()
}

fn print_controls() {
    let quit = if cfg!(target_os = "macos") { "Cmd-Q" } else { "Ctrl-Q" };
    println!(
        "\ncontrols (Google Earth style, keyboard only):\n  \
         Arrow keys      pan across the surface\n  \
         W / S  (+ / -)  zoom in / out\n  \
         A / D           rotate (spin) the view\n  \
         Q / E           tilt (top-down <-> horizon)\n  \
         Shift           move faster (hold)\n  \
         T               guided tour (relaxing autopilot)\n  \
         R               teleport to a random spot\n  \
         P               print location & seed\n  \
         G               toggle wireframe\n  \
         Esc             graphics settings + help (arrows adjust)\n  \
         {quit} / close  quit\n"
    );
}

/// Options for the headless `--video` recorder (parsed by [`parse_video`]).
struct VideoOptions {
    path: PathBuf,
    width: u32,
    height: u32,
    fps: u32,
    /// Auto-stop after this many recorded seconds; `None` = run until Ctrl+C.
    max_seconds: Option<u32>,
    /// Detail preset to record at (defaults to [`VIDEO_PRESET`]).
    graphics: settings::Graphics,
    /// Record without the background-music track (`--video-mute`).
    mute: bool,
}

/// Parse `--video` and its options, or `None` if not recording. Tolerant of bad
/// values (warn + fall back to defaults), like the other CLI parsers.
fn parse_video() -> Option<VideoOptions> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.iter().any(|a| a == "--video") {
        return None;
    }
    let mut path: Option<PathBuf> = None;
    let (mut width, mut height, mut fps) = (VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS);
    let mut max_seconds = None;
    let mut graphics = settings::Graphics::from_preset(VIDEO_PRESET)
        .expect("VIDEO_PRESET must name a real preset tier");
    let mut mute = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--video-out" => {
                if let Some(v) = it.next() {
                    path = Some(PathBuf::from(v));
                }
            }
            "--video-size" => {
                if let Some(v) = it.next() {
                    match v
                        .split_once(['x', 'X'])
                        .and_then(|(a, b)| Some((a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?)))
                    {
                        Some((w, h)) => {
                            width = w;
                            height = h;
                        }
                        None => eprintln!("warning: bad --video-size '{v}', using {VIDEO_WIDTH}x{VIDEO_HEIGHT}"),
                    }
                }
            }
            "--video-fps" => {
                if let Some(v) = it.next() {
                    match v.parse::<u32>() {
                        Ok(n) => fps = n,
                        Err(_) => eprintln!("warning: bad --video-fps '{v}', using {VIDEO_FPS}"),
                    }
                }
            }
            "--video-seconds" => {
                if let Some(v) = it.next() {
                    match v.parse::<u32>() {
                        Ok(n) => max_seconds = Some(n),
                        Err(_) => eprintln!("warning: bad --video-seconds '{v}', recording until Ctrl+C"),
                    }
                }
            }
            "--video-preset" => {
                if let Some(v) = it.next() {
                    match settings::Graphics::from_preset(v) {
                        Some(g) => graphics = g,
                        None => eprintln!(
                            "warning: unknown --video-preset '{v}', using {VIDEO_PRESET} (choices: {})",
                            settings::Graphics::preset_names().join(", ")
                        ),
                    }
                }
            }
            "--video-mute" => mute = true,
            _ => {}
        }
    }
    // yuv420p needs even dimensions; keep at least 2 px and ≥ 1 fps.
    width = (width & !1).max(2);
    height = (height & !1).max(2);
    fps = fps.max(1);
    Some(VideoOptions { path: path.unwrap_or_else(default_video_path), width, height, fps, max_seconds, graphics, mute })
}

/// Where `--video` writes by default: the shared drop point on macOS (reachable from
/// the GUI account, like the log and the app bundle), the OS temp dir elsewhere.
fn default_video_path() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        PathBuf::from("/Users/Shared/planet-explorer-tour.mp4")
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::env::temp_dir().join("planet-explorer-tour.mp4")
    }
}

/// Headless tour recorder: build a surfaceless renderer, fly the guided tour at a
/// fixed timestep, and pipe each rendered frame to ffmpeg as an MP4. Ctrl+C stops
/// the loop and finalizes the file. No window or event loop is created, so it runs
/// on a headless box.
fn run_video(seed: u64, planet: Arc<Planet>, units: units::Units, opts: VideoOptions) -> anyhow::Result<()> {
    let (w, h, fps) = (opts.width, opts.height, opts.fps);

    // Ctrl+C → stop flag: the loop breaks and we finalize, so the file stays playable.
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst)).context("installing Ctrl+C handler")?;
    }

    // Background music: the same shuffled-playlist idea as the live app, but seeded
    // from the planet seed so a given world always gets the same soundtrack. Muxed
    // in by the encoder; `--video-mute` records silent.
    let soundtrack = if opts.mute { Vec::new() } else { audio::shuffled_soundtrack(seed) };

    // Start ffmpeg FIRST so a missing/broken encoder fails fast, before any GPU work.
    let mut encoder =
        video::VideoEncoder::start(&opts.path, w, h, fps, &soundtrack).context("starting the video encoder")?;

    // Headless renderer + streamer — mirrors App::resumed, minus window and audio.
    // Video isn't real-time bound, so we record at a high preset regardless of the
    // host GPU: the live frame rate doesn't matter (we encode a fixed timestep), and
    // per-frame settling lets the streamer fully resolve each frame. The preset is
    // chosen by --video-preset (default VIDEO_PRESET).
    let graphics = opts.graphics;
    let mesh_cfg = mesh::MeshConfig::new(graphics.mesh_res, graphics.veg_min_level, graphics.veg_density);
    let flora = &planet.flora;
    let mut renderer = pollster::block_on(Renderer::new_offscreen(w, h, flora.meshes(), flora.textures())).context("creating the headless renderer")?;

    let mut camera = Camera::new(&planet, Vec3::new(0.4, 0.5, 0.77).normalize());
    camera.set_aspect(w, h);

    // Seed the six root chunks so the first frame is never blank.
    for root in lod::ChunkKey::roots() {
        let cpu = mesh::CpuChunk::build(&planet, root, &mesh_cfg);
        renderer.upload_chunk(root, cpu);
    }
    let threads = std::thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(3).max(1);
    let mut streamer = Streamer::new(planet.clone(), threads, mesh_cfg.clone());

    let mut tour = tour::Tour::new(&camera, &planet);
    let dt = 1.0 / fps as f32;
    let max_frames = opts.max_seconds.map(|s| s as u64 * fps as u64);

    println!("recording {w}x{h} @ {fps}fps ({} detail) -> {}", graphics.preset, opts.path.display());
    match opts.max_seconds {
        Some(s) => println!("  stopping after {s}s (Ctrl+C to stop sooner)"),
        None => println!("  press Ctrl+C to stop and finalize the file"),
    }
    match soundtrack.len() {
        0 => println!("  audio: muted"),
        n => println!("  audio: {n} shuffled tracks (seeded from world)"),
    }
    info!(w, h, fps, detail = graphics.preset, tracks = soundtrack.len(), path = %opts.path.display(), seconds = ?opts.max_seconds, "video recording started");

    let mut frame_idx: u64 = 0;
    let mut sim_time = 0.0f32;
    let mut last_phase = "";
    let mut toured = false; // set when the tour has cruised every biome → play the finale
    loop {
        if stop.load(Ordering::SeqCst) {
            println!("\nCtrl+C received — finalizing…");
            break;
        }
        if let Some(mf) = max_frames
            && frame_idx >= mf
        {
            println!("\nreached target length — finalizing…");
            break;
        }

        // Advance the tour with a fixed timestep (smooth, render-speed independent).
        tour.update(dt, &planet, &mut camera);
        sim_time += dt;

        let rgba = capture_frame(&mut renderer, &mut streamer, &planet, &camera, &graphics, seed, sim_time);
        encoder.write_frame(&rgba).context("piping a frame to ffmpeg")?;
        frame_idx += 1;

        // Progress on stdout: a line at each phase change, plus a periodic heartbeat,
        // so the user can follow the tour and decide when to Ctrl+C.
        let phase = tour.phase_label();
        let heartbeat = frame_idx.is_multiple_of(fps as u64 * VIDEO_LOG_EVERY_SECONDS);
        if phase != last_phase || heartbeat {
            let (lat, lon) = camera.lat_lon();
            let biome = planet.sample(camera.focus).biome.name();
            println!(
                "  [{sim_time:>6.1}s] {phase:<7} {biome:<16} {lat:>6.1}°,{lon:>7.1}°  alt {}  ({frame_idx} frames)",
                units::distance(camera.altitude(), units)
            );
            last_phase = phase;
        }

        // Once the tour has cruised every biome in its cycle, stop and run the finale.
        if tour.toured_all_biomes() {
            toured = true;
            break;
        }
    }

    // --- Finale: pull back to the whole planet from space, rotate it slowly, then
    // fade the video to black (and the audio to silence). Only when the tour finished
    // on its own — a Ctrl+C or a --video-seconds cap mid-tour just finalizes as-is.
    if toured && !stop.load(Ordering::SeqCst) {
        println!("\n  tour complete — finale: pulling back to the whole planet");
        info!("video finale started");
        let (f0, d0, t0, h0) = (camera.focus, camera.distance(), camera.tilt(), camera.heading());
        let space_dist = FINALE_SPACE_DISTANCE_FACTOR * planet::PLANET_RADIUS;
        let finale_frames = ((FINALE_ROTATE_SECONDS + FINALE_FADE_SECONDS) * fps as f32).round() as u64;
        for i in 0..finale_frames {
            if stop.load(Ordering::SeqCst) {
                println!("\nCtrl+C received — finalizing…");
                break;
            }
            if let Some(mf) = max_frames
                && frame_idx >= mf
            {
                break;
            }
            let ft = i as f32 * dt; // seconds into the finale
            // Pull back: ease the focus distance out to the space view and the tilt up
            // to top-down (exponential zoom feels natural across the scale change).
            let p = planet::smoothstep(0.0, 1.0, (ft / FINALE_PULLBACK_SECONDS).min(1.0));
            let dist = d0 * (space_dist / d0).powf(p);
            let tilt = t0 * (1.0 - p);
            // Slow, continuous spin of the globe about its axis: rotate the sub-camera
            // point around +Y; the eye sits straight above it and orbits with it.
            let focus = glam::Quat::from_axis_angle(Vec3::Y, FINALE_ROTATE_RATE * ft) * f0;
            camera.set_view(focus, dist, h0, tilt);
            sim_time += dt;

            let mut rgba = capture_frame(&mut renderer, &mut streamer, &planet, &camera, &graphics, seed, sim_time);
            // Fade to black over the last FINALE_FADE_SECONDS, still rotating.
            let fade_t = ft - FINALE_ROTATE_SECONDS;
            if fade_t > 0.0 {
                fade_to_black(&mut rgba, 1.0 - fade_t / FINALE_FADE_SECONDS);
            }
            encoder.write_frame(&rgba).context("piping a finale frame to ffmpeg")?;
            frame_idx += 1;

            if frame_idx.is_multiple_of(fps as u64 * VIDEO_LOG_EVERY_SECONDS) {
                let stage = if fade_t > 0.0 { "fading" } else { "rotating" };
                println!(
                    "  [{sim_time:>6.1}s] Finale  {stage:<8}  ({frame_idx} frames)",
                );
            }
        }
    }

    // Fade the music out under the finale's video fade (same trailing window).
    let audio_fade = (toured && !soundtrack.is_empty()).then(|| video::AudioFadeOut {
        start_secs: (sim_time - FINALE_FADE_SECONDS).max(0.0),
        dur_secs: FINALE_FADE_SECONDS,
    });
    let path = encoder.finish(audio_fade).context("finalizing the video")?;
    println!("video written: {}", path.display());
    info!(path = %path.display(), frames = frame_idx, seconds = round1(sim_time), "video finalized");
    Ok(())
}

/// Settle the streamer for the camera's current view (bounded by VIDEO_SETTLE_MAX_MS),
/// update the per-frame globals, and render one frame to RGBA — the shared per-frame
/// capture for both the tour and the finale in [`run_video`].
fn capture_frame(
    renderer: &mut Renderer,
    streamer: &mut Streamer,
    planet: &Planet,
    camera: &Camera,
    graphics: &settings::Graphics,
    seed: u64,
    sim_time: f32,
) -> Vec<u8> {
    // Let the streamer load the visible terrain/veg before we capture (bounded so a
    // chunk that never meshes can't hang a frame).
    let cam_pos = camera.position(planet);
    let settle = Instant::now();
    let sel = loop {
        for (key, cpu) in streamer.poll() {
            renderer.upload_chunk(key, cpu);
        }
        let sel = lod::select(planet, cam_pos, graphics.split_factor, &|k| renderer.has_chunk(k));
        if sel.want.is_empty() || settle.elapsed().as_millis() >= VIDEO_SETTLE_MAX_MS {
            break sel;
        }
        for key in &sel.want {
            streamer.request(*key);
        }
        std::thread::sleep(std::time::Duration::from_millis(VIDEO_SETTLE_SLEEP_MS));
    };

    let keep: std::collections::HashSet<_> = sel.draw.iter().copied().collect();
    renderer.evict(&keep, graphics.mem_budget_bytes());

    // Per-frame uniforms — identical to App::frame's globals.
    let (view_proj, _view, pos) = camera.view_proj(planet);
    let inv = view_proj.inverse();
    let globals = Globals {
        view_proj: view_proj.to_cols_array_2d(),
        inv_view_proj: inv.to_cols_array_2d(),
        camera_pos: [pos.x, pos.y, pos.z, sim_time],
        sun_dir: [planet.sun_dir.x, planet.sun_dir.y, planet.sun_dir.z, SUN_AMBIENT],
        params: [camera.fog_density(), planet::PLANET_RADIUS, planet::SEA_LEVEL, camera.altitude()],
        atmosphere: [planet.atmosphere.x, planet.atmosphere.y, planet.atmosphere.z, (seed % 997) as f32],
    };
    renderer.update_globals(&globals);
    renderer.render_to_rgba(&sel.draw)
}

/// Scale a frame's RGB toward black by `brightness` (1 = unchanged, 0 = black),
/// leaving alpha. Used for the finale fade-out.
fn fade_to_black(rgba: &mut [u8], brightness: f32) {
    let b = brightness.clamp(0.0, 1.0);
    for px in rgba.chunks_exact_mut(4) {
        px[0] = (px[0] as f32 * b) as u8;
        px[1] = (px[1] as f32 * b) as u8;
        px[2] = (px[2] as f32 * b) as u8;
    }
}

struct App {
    seed: u64,
    planet: Arc<Planet>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    streamer: Option<Streamer>,
    camera: Camera,
    start: Instant,
    last: Instant,
    title_timer: f32,
    mods: ModifiersState,
    audio: Option<audio::Audio>,
    units: units::Units,
    tour: Option<tour::Tour>,
    /// Set once the user presses any key; gates the launch auto-tour (attract mode).
    had_input: bool,

    // Graphics settings + the ESC settings menu.
    graphics: settings::Graphics,
    mesh_cfg: Arc<mesh::MeshConfig>,
    menu_open: bool,
    menu_tab: usize,
    menu_sel: usize,
    /// Rebuild-relevant settings captured when the menu opened, to detect change.
    menu_open_sig: (u32, u32, u32),

    // Performance sampling (aggregated, logged at DEBUG every couple seconds).
    perf_accum: f32,
    perf_frames: u32,
    frame_ms_max: f32,
    uploads_period: u32,
    last_hitch: f32,
}

impl App {
    fn new(seed: u64, planet: Arc<Planet>, units: units::Units) -> Self {
        // Start in orbit above a pleasant mid-latitude.
        let anchor = Vec3::new(0.4, 0.5, 0.77).normalize();
        let camera = Camera::new(&planet, anchor);
        let graphics = settings::Graphics::default();
        let mesh_cfg = mesh::MeshConfig::new(graphics.mesh_res, graphics.veg_min_level, graphics.veg_density);
        Self {
            seed,
            planet,
            window: None,
            renderer: None,
            streamer: None,
            camera,
            start: Instant::now(),
            last: Instant::now(),
            title_timer: 0.0,
            mods: ModifiersState::empty(),
            audio: None,
            units,
            tour: None,
            had_input: false,
            graphics,
            mesh_cfg,
            menu_open: false,
            menu_tab: settings::TAB_HELP,
            menu_sel: 0,
            menu_open_sig: (0, 0, 0),
            perf_accum: 0.0,
            perf_frames: 0,
            frame_ms_max: 0.0,
            uploads_period: 0,
            last_hitch: 0.0,
        }
    }

    fn frame(&mut self) {
        let (Some(renderer), Some(streamer)) = (self.renderer.as_mut(), self.streamer.as_mut()) else {
            return;
        };
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.1);
        self.last = now;
        let time = (now - self.start).as_secs_f32();

        // Attract mode: if the user hasn't touched anything for a few seconds after
        // launch, start the guided tour automatically.
        if !self.had_input && self.tour.is_none() && time >= AUTO_TOUR_IDLE_SECONDS {
            self.tour = Some(tour::Tour::new(&self.camera, &self.planet));
            info!(action = "tour", "auto-started guided tour (idle at launch)");
        }

        // Advance the playlist (reshuffles when a round finishes) and the camera.
        if let Some(audio) = &mut self.audio {
            audio.tick();
        }
        // The guided tour, when active, flies the camera itself; otherwise the
        // player drives it. ("Move faster" comes from the authoritative modifier
        // state — not Shift key events, which can get stuck on at launch.)
        if let Some(tour) = &mut self.tour {
            tour.update(dt, &self.planet, &mut self.camera);
        } else if !self.menu_open {
            // The camera is frozen while the settings menu is open (the arrow keys
            // drive the menu instead).
            self.camera.key(KeyAction::Boost, self.mods.shift_key());
            self.camera.update(dt, &self.planet);
        }

        let polled = streamer.poll();
        let uploads = polled.len();
        for (key, cpu) in polled {
            tracing::trace!(?key, verts = cpu.vertices.len(), veg_instances = cpu.veg.instances.len(), "chunk uploaded");
            renderer.upload_chunk(key, cpu);
        }

        let cam_pos = self.camera.position(&self.planet);
        let sel = lod::select(&self.planet, cam_pos, self.graphics.split_factor, &|k| renderer.has_chunk(k));
        let draw_count = sel.draw.len();

        // Request the nearest wanted chunks first.
        let mut want = sel.want;
        // Decorate-sort: `surface_radius` samples the (expensive) height noise, so
        // compute each chunk's squared camera distance exactly once per key instead
        // of twice per comparison — O(n) noise samples per frame, not O(n log n).
        // Distance-squared is non-negative, so its IEEE-754 bits order identically
        // to the value, giving an integer sort key without needing `f32: Ord`.
        want.sort_by_cached_key(|k| {
            let d = k.center_dir();
            (d * self.planet.surface_radius(d) - cam_pos).length_squared().to_bits()
        });
        for key in want.into_iter().take(MAX_REQUESTS_PER_FRAME) {
            streamer.request(key);
        }

        let keep: std::collections::HashSet<_> = sel.draw.iter().copied().collect();
        renderer.evict(&keep, self.graphics.mem_budget_bytes());

        // Assemble the per-frame uniforms.
        let (view_proj, _view, pos) = self.camera.view_proj(&self.planet);
        let inv = view_proj.inverse();
        let globals = Globals {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv.to_cols_array_2d(),
            camera_pos: [pos.x, pos.y, pos.z, time],
            sun_dir: [self.planet.sun_dir.x, self.planet.sun_dir.y, self.planet.sun_dir.z, SUN_AMBIENT],
            params: [self.camera.fog_density(), planet::PLANET_RADIUS, planet::SEA_LEVEL, self.camera.altitude()],
            atmosphere: [self.planet.atmosphere.x, self.planet.atmosphere.y, self.planet.atmosphere.z, (self.seed % 997) as f32],
        };
        renderer.update_globals(&globals);

        renderer.render(&sel.draw);

        // --- performance sampling -----------------------------------------
        let frame_ms = dt * 1000.0;
        self.perf_frames += 1;
        self.perf_accum += dt;
        self.frame_ms_max = self.frame_ms_max.max(frame_ms);
        self.uploads_period += uploads as u32;

        // A single slow frame is worth flagging immediately (rate-limited), so a
        // stutter during testing is easy to find in the log.
        if frame_ms > FRAME_HITCH_MS && time - self.last_hitch > HITCH_LOG_COOLDOWN {
            self.last_hitch = time;
            warn!(
                target: "perf",
                frame_ms = round1(frame_ms),
                draw = draw_count,
                uploads,
                pending = streamer.pending_count(),
                alt = units::distance(self.camera.altitude(), self.units),
                "frame hitch"
            );
        }

        // Aggregate sample every ~2s at DEBUG: the spine of perf analysis.
        if self.perf_accum >= PERF_SAMPLE_SECONDS {
            let fps = self.perf_frames as f32 / self.perf_accum;
            let avg_ms = self.perf_accum * 1000.0 / self.perf_frames as f32;
            let (lat, lon) = self.camera.lat_lon();
            let biome = self.planet.sample(self.camera.focus).biome.name();
            debug!(
                target: "perf",
                fps = round1(fps),
                avg_ms = round1(avg_ms),
                max_ms = round1(self.frame_ms_max),
                alt = units::distance(self.camera.altitude(), self.units),
                lat = round1(lat),
                lon = round1(lon),
                biome,
                chunks = renderer.chunk_count(),
                mem_mb = round1(renderer.resident_bytes() as f32 / BYTES_PER_MIB),
                draw = draw_count,
                pending = streamer.pending_count(),
                uploads = self.uploads_period,
                "perf"
            );
            self.perf_accum = 0.0;
            self.perf_frames = 0;
            self.frame_ms_max = 0.0;
            self.uploads_period = 0;
        }

        // Throttled window-title HUD.
        self.title_timer += dt;
        if self.title_timer > TITLE_UPDATE_SECONDS {
            self.title_timer = 0.0;
            let (lat, lon) = self.camera.lat_lon();
            let biome = self.planet.sample(self.camera.focus).biome.name();
            let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
            if let Some(w) = &self.window {
                w.set_title(&format!(
                    "planet-explorer — seed {} | {:.0} fps | alt {} | {:.1}°,{:.1}° | {} | chunks {} | {:.0} MB",
                    self.seed,
                    fps,
                    units::distance(self.camera.altitude(), self.units),
                    lat, lon, biome, renderer.chunk_count(),
                    renderer.resident_bytes() as f32 / BYTES_PER_MIB
                ));
            }
        }
    }

    fn print_location(&self) {
        let (lat, lon) = self.camera.lat_lon();
        let s = self.planet.sample(self.camera.focus);
        let alt = units::distance(self.camera.altitude(), self.units);
        let terrain = units::elevation(s.height, self.units);
        println!(
            "location: seed {} | lat {:.3}° lon {:.3}° | altitude {} | terrain {} | biome {}",
            self.seed, lat, lon, alt, terrain, s.biome.name()
        );
        info!(
            seed = self.seed,
            lat = round1(lat),
            lon = round1(lon),
            altitude = alt,
            terrain = terrain,
            biome = s.biome.name(),
            "location"
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(format!("planet-explorer — seed {}", self.seed))
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            // Start filling the screen (borderless fullscreen) — the user always
            // expands the window first thing. `with_inner_size` is the fallback if
            // they drop out of fullscreen.
            .with_fullscreen(Some(Fullscreen::Borderless(None)));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let mut renderer = {
            // Archetype base meshes + shared texture array for instanced vegetation
            // (uploaded once). See crate::models / crate::flora.
            let flora = &self.planet.flora;
            pollster::block_on(Renderer::new(window.clone(), flora.meshes(), flora.textures())).expect("renderer init")
        };
        self.camera.set_aspect(renderer.size.0, renderer.size.1);
        let (lines, hl) = overlay::menu(&self.graphics, self.menu_tab, self.menu_sel, self.units);
        renderer.set_overlay(lines, hl);

        // Generate the six root chunks up front so there's always a planet to see.
        for root in lod::ChunkKey::roots() {
            let cpu = mesh::CpuChunk::build(&self.planet, root, &self.mesh_cfg);
            renderer.upload_chunk(root, cpu);
        }

        let threads = std::thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(3).max(1);
        let streamer = Streamer::new(self.planet.clone(), threads, self.mesh_cfg.clone());

        info!(
            window_size = ?(renderer.size.0, renderer.size.1),
            worker_threads = threads,
            wireframe_supported = renderer.supports_wireframe,
            "renderer ready; entering main loop"
        );

        // Start the looping soundtrack (non-fatal if there's no audio device).
        if self.audio.is_none() {
            self.audio = audio::Audio::start(AUDIO_VOLUME);
        }

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.streamer = Some(streamer);
        self.last = Instant::now();
        // Count the attract-mode idle window from when the app is actually up
        // (renderer/window init can take a moment).
        self.start = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("close requested; exiting");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                debug!(w = size.width, h = size.height, "resized");
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
                self.camera.set_aspect(size.width, size.height);
            }
            WindowEvent::RedrawRequested => self.frame(),
            // Keyboard-only: mouse buttons, motion, and wheel are intentionally ignored.
            WindowEvent::ModifiersChanged(m) => self.mods = m.state(),
            WindowEvent::KeyboardInput { event, is_synthetic, .. } => self.handle_key(event_loop, event, is_synthetic),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl App {
    /// Push the current settings/selection into the overlay geometry.
    fn refresh_overlay(&mut self) {
        let (lines, hl) = overlay::menu(&self.graphics, self.menu_tab, self.menu_sel, self.units);
        if let Some(r) = &mut self.renderer {
            r.set_overlay(lines, hl);
        }
    }

    fn open_menu(&mut self) {
        self.menu_open = true;
        self.menu_open_sig = self.graphics.rebuild_signature();
        self.camera.release_keys(); // don't keep drifting on a held key
        self.refresh_overlay();
        if let Some(r) = &mut self.renderer {
            r.set_overlay_visible(true);
        }
        info!(action = "menu", preset = self.graphics.preset, "graphics menu opened");
    }

    fn close_menu(&mut self) {
        self.menu_open = false;
        if let Some(r) = &mut self.renderer {
            r.set_overlay_visible(false);
        }
        // Settings baked into geometry only take effect on a rebuild.
        if self.graphics.rebuild_signature() != self.menu_open_sig {
            self.apply_rebuild();
        }
        info!(action = "menu", preset = self.graphics.preset, "graphics menu closed");
    }

    fn menu_switch_tab(&mut self) {
        self.menu_tab = (self.menu_tab + 1) % settings::TAB_COUNT;
        self.refresh_overlay();
    }

    fn menu_move(&mut self, dir: i32) {
        let n = settings::ROW_COUNT as i32;
        self.menu_sel = (self.menu_sel as i32 + dir).rem_euclid(n) as usize;
        self.refresh_overlay();
    }

    fn menu_adjust(&mut self, dir: i32) {
        self.graphics.adjust(self.menu_sel, dir);
        // Terrain detail and memory budget are read fresh every frame, so they
        // apply live; mesh/vegetation changes are applied on close (they rebuild).
        self.refresh_overlay();
    }

    /// Re-mesh the world at the current mesh/vegetation settings.
    fn apply_rebuild(&mut self) {
        let g = self.graphics;
        self.mesh_cfg.set(g.mesh_res, g.veg_min_level, g.veg_density);
        if let Some(s) = &mut self.streamer {
            s.clear();
        }
        if let Some(r) = &mut self.renderer {
            r.clear_chunks();
            // Rebuild the roots immediately so there's never a blank frame.
            for root in lod::ChunkKey::roots() {
                let cpu = mesh::CpuChunk::build(&self.planet, root, &self.mesh_cfg);
                r.upload_chunk(root, cpu);
            }
        }
        info!(
            grid = g.mesh_res,
            veg_min_level = g.veg_min_level,
            veg_density = g.veg_density,
            "rebuilding world at new mesh/vegetation settings"
        );
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, ev: winit::event::KeyEvent, is_synthetic: bool) {
        let pressed = ev.state == ElementState::Pressed;
        // A real keypress cancels attract mode. Ignore *synthetic* key events: when
        // a window gains focus, winit replays currently-held keys as synthetic
        // presses — notably on Windows as the borderless-fullscreen window grabs
        // focus at launch. Counting those cancelled the launch auto-tour before it
        // could start (the bug was Windows-only; macOS doesn't emit these).
        if pressed && !is_synthetic {
            self.had_input = true;
        }
        let PhysicalKey::Code(code) = ev.physical_key else { return };

        // While the settings menu is open the arrow keys drive it (not the
        // camera); Esc closes it. Auto-repeat is allowed for the sliders.
        if self.menu_open {
            if pressed {
                let on_graphics = self.menu_tab == settings::TAB_GRAPHICS;
                match code {
                    KeyCode::Escape if !ev.repeat => self.close_menu(),
                    KeyCode::Tab if !ev.repeat => self.menu_switch_tab(),
                    // Settings navigation only applies on the GRAPHICS tab.
                    KeyCode::ArrowUp if on_graphics => self.menu_move(-1),
                    KeyCode::ArrowDown if on_graphics => self.menu_move(1),
                    KeyCode::ArrowLeft if on_graphics => self.menu_adjust(-1),
                    KeyCode::ArrowRight if on_graphics => self.menu_adjust(1),
                    KeyCode::KeyQ if self.mods.super_key() || self.mods.control_key() => {
                        info!("quit shortcut; exiting");
                        event_loop.exit();
                    }
                    _ => {}
                }
            }
            return;
        }

        // Continuous controls — Google Earth style: arrows pan, W/S zoom,
        // A/D rotate, Q/E tilt, Shift to move faster.
        let action = match code {
            KeyCode::ArrowUp => Some(KeyAction::PanForward),
            KeyCode::ArrowDown => Some(KeyAction::PanBack),
            KeyCode::ArrowLeft => Some(KeyAction::PanLeft),
            KeyCode::ArrowRight => Some(KeyAction::PanRight),
            KeyCode::KeyW | KeyCode::Equal | KeyCode::NumpadAdd => Some(KeyAction::ZoomIn),
            KeyCode::KeyS | KeyCode::Minus | KeyCode::NumpadSubtract => Some(KeyAction::ZoomOut),
            KeyCode::KeyA => Some(KeyAction::RotateLeft),
            KeyCode::KeyD => Some(KeyAction::RotateRight),
            KeyCode::KeyE => Some(KeyAction::TiltMore),
            KeyCode::KeyQ => Some(KeyAction::TiltLess),
            // Shift (boost) is handled via modifier state in frame(), not here.
            _ => None,
        };
        if let Some(a) = action {
            // Taking manual control ends the guided tour.
            if pressed && self.tour.take().is_some() {
                info!("tour stopped (manual control)");
            }
            self.camera.key(a, pressed);
            return;
        }

        // One-shot actions (on key-down, ignoring auto-repeat).
        if !pressed || ev.repeat {
            return;
        }

        // Quit: Cmd+Q (macOS) or Ctrl+Q (Windows/Linux). The window close button
        // and the app menu's Quit also work everywhere.
        if code == KeyCode::KeyQ && (self.mods.super_key() || self.mods.control_key()) {
            info!("quit shortcut; exiting");
            event_loop.exit();
            return;
        }

        match code {
            KeyCode::Escape => self.open_menu(),
            KeyCode::KeyR => {
                self.camera.teleport(&self.planet, random_unit());
                let (lat, lon) = self.camera.lat_lon();
                info!(action = "teleport", lat = round1(lat), lon = round1(lon), "teleported to random surface point");
                self.print_location();
            }
            KeyCode::KeyP => self.print_location(),
            KeyCode::KeyT => {
                if self.tour.take().is_some() {
                    info!(action = "tour", "guided tour stopped");
                } else {
                    self.tour = Some(tour::Tour::new(&self.camera, &self.planet));
                    info!(action = "tour", "guided tour started");
                }
            }
            KeyCode::KeyG => {
                if let Some(r) = &mut self.renderer {
                    if r.supports_wireframe {
                        r.wireframe = !r.wireframe;
                        debug!(action = "wireframe", on = r.wireframe, "wireframe toggled");
                    } else {
                        debug!(action = "wireframe", "wireframe unsupported on this adapter");
                        println!("wireframe not supported on this adapter");
                    }
                }
            }
            _ => {}
        }
    }
}

/// Round to one decimal place for tidy log fields.
fn round1(x: f32) -> f32 {
    (x * 10.0).round() / 10.0
}

fn random_unit() -> Vec3 {
    // Uniform point on the sphere.
    let z = rand::random::<f32>() * 2.0 - 1.0;
    let a = rand::random::<f32>() * std::f32::consts::TAU;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * a.cos(), z, r * a.sin())
}

