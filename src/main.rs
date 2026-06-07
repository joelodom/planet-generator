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
mod font8x8;
mod gfx;
mod lod;
mod logging;
mod mesh;
mod overlay;
mod planet;
mod units;
#[cfg(test)]
mod tests;

use camera::{Camera, KeyAction};
use gfx::{Globals, Renderer};
use glam::Vec3;
use lod::Streamer;
use planet::Planet;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

const SUN_AMBIENT: f32 = 0.32;
/// Cap chunk requests per frame so a fast camera can't flood the work queue.
const MAX_REQUESTS_PER_FRAME: usize = 48;
/// Soft limit on resident GPU chunks before far ones get evicted.
const CHUNK_CACHE_LIMIT: usize = 1800;

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
    print_controls();

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
        if a == "--units" {
            if let Some(v) = args.next() {
                if let Some(u) = units::Units::parse(&v) {
                    return u;
                }
                eprintln!("warning: unknown --units '{v}', using metric");
            }
        }
    }
    units::Units::Metric
}

fn parse_seed() -> u64 {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--seed" || a == "-s" {
            if let Some(v) = args.next() {
                if let Ok(n) = v.parse::<u64>() {
                    return n;
                }
                eprintln!("warning: could not parse seed '{v}', using random");
            }
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
         R               teleport to a random spot\n  \
         P               print location & seed\n  \
         G               toggle wireframe\n  \
         Esc             toggle help overlay\n  \
         {quit} / close  quit\n"
    );
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

        // Advance the playlist (reshuffles when a round finishes) and the camera.
        if let Some(audio) = &mut self.audio {
            audio.tick();
        }
        self.camera.update(dt, &self.planet);

        let polled = streamer.poll();
        let uploads = polled.len();
        for (key, cpu) in polled {
            tracing::trace!(?key, verts = cpu.vertices.len(), trees = cpu.trees.len(), "chunk uploaded");
            renderer.upload_chunk(key, cpu);
        }

        let cam_pos = self.camera.position(&self.planet);
        let sel = lod::select(&self.planet, cam_pos, &|k| renderer.has_chunk(k));
        let draw_count = sel.draw.len();

        // Request the nearest wanted chunks first.
        let mut want = sel.want;
        want.sort_by(|a, b| {
            let da = (a.center_dir() * self.planet.surface_radius(a.center_dir()) - cam_pos).length_squared();
            let db = (b.center_dir() * self.planet.surface_radius(b.center_dir()) - cam_pos).length_squared();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        for key in want.into_iter().take(MAX_REQUESTS_PER_FRAME) {
            streamer.request(key);
        }

        let keep: std::collections::HashSet<_> = sel.draw.iter().copied().collect();
        renderer.evict(&keep, CHUNK_CACHE_LIMIT);

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
        if frame_ms > 120.0 && time - self.last_hitch > 1.0 {
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
        if self.perf_accum >= 2.0 {
            let fps = self.perf_frames as f32 / self.perf_accum;
            let avg_ms = self.perf_accum * 1000.0 / self.perf_frames as f32;
            let (lat, lon) = self.camera.lat_lon();
            let biome = biome_name(self.planet.sample(self.camera.focus).biome);
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
        if self.title_timer > 0.4 {
            self.title_timer = 0.0;
            let (lat, lon) = self.camera.lat_lon();
            let biome = biome_name(self.planet.sample(self.camera.focus).biome);
            let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
            if let Some(w) = &self.window {
                w.set_title(&format!(
                    "planet-explorer — seed {} | {:.0} fps | alt {} | {:.1}°,{:.1}° | {} | chunks {}",
                    self.seed,
                    fps,
                    units::distance(self.camera.altitude(), self.units),
                    lat, lon, biome, renderer.chunk_count()
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
            self.seed, lat, lon, alt, terrain, biome_name(s.biome)
        );
        info!(
            seed = self.seed,
            lat = round1(lat),
            lon = round1(lon),
            altitude = alt,
            terrain = terrain,
            biome = biome_name(s.biome),
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
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let mut renderer = pollster::block_on(Renderer::new(window.clone())).expect("renderer init");
        self.camera.set_aspect(renderer.size.0, renderer.size.1);
        renderer.set_overlay_lines(overlay::help_lines());

        // Generate the six root chunks up front so there's always a planet to see.
        for root in lod::ChunkKey::roots() {
            let cpu = mesh::CpuChunk::build(&self.planet, root);
            renderer.upload_chunk(root, cpu);
        }

        let threads = std::thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(3).max(1);
        let streamer = Streamer::new(self.planet.clone(), threads);

        info!(
            window_size = ?(renderer.size.0, renderer.size.1),
            worker_threads = threads,
            wireframe_supported = renderer.supports_wireframe,
            "renderer ready; entering main loop"
        );

        // Start the looping soundtrack (non-fatal if there's no audio device).
        if self.audio.is_none() {
            self.audio = audio::Audio::start(0.5);
        }

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.streamer = Some(streamer);
        self.last = Instant::now();
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
            WindowEvent::KeyboardInput { event, .. } => self.handle_key(event_loop, event),
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
    fn handle_key(&mut self, event_loop: &ActiveEventLoop, ev: winit::event::KeyEvent) {
        let pressed = ev.state == ElementState::Pressed;
        let PhysicalKey::Code(code) = ev.physical_key else { return };

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
            KeyCode::ShiftLeft | KeyCode::ShiftRight => Some(KeyAction::Boost),
            _ => None,
        };
        if let Some(a) = action {
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
            KeyCode::Escape => {
                debug!(action = "help", "toggled help overlay");
                if let Some(r) = &mut self.renderer {
                    r.toggle_overlay();
                }
            }
            KeyCode::KeyR => {
                self.camera.teleport(&self.planet, random_unit());
                let (lat, lon) = self.camera.lat_lon();
                info!(action = "teleport", lat = round1(lat), lon = round1(lon), "teleported to random surface point");
                self.print_location();
            }
            KeyCode::KeyP => self.print_location(),
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

fn biome_name(b: planet::Biome) -> &'static str {
    use planet::Biome::*;
    match b {
        Ocean => "ocean",
        Beach => "beach",
        PolarIce => "polar ice",
        Tundra => "tundra",
        BorealForest => "boreal forest",
        Grassland => "grassland",
        TemperateForest => "temperate forest",
        Desert => "desert",
        TropicalForest => "tropical forest",
        Mountain => "mountain",
        Snow => "snow",
    }
}
