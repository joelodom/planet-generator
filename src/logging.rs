//! Logging setup.
//!
//! Uses the `tracing` ecosystem (the modern idiomatic choice for Rust apps) and
//! writes to a **single file, always appended** across restarts, so a whole
//! testing history accumulates in one place. The file is the primary artifact:
//! this app is typically launched on another macOS account, and the log is how
//! we diagnose bugs and performance after the fact — so every line is absolute-
//! timestamped and carries its module target and thread name.
//!
//! Levels follow convention:
//!   ERROR — something failed (render error, device loss, panic)
//!   WARN  — recoverable trouble or a performance hitch worth noticing
//!   INFO  — lifecycle milestones (startup, GPU selected, teleport, exit)
//!   DEBUG — periodic performance samples and user actions (the default floor)
//!   TRACE — per-chunk / per-frame firehose (off unless RUST_LOG asks for it)
//!
//! Default filter: our crate at DEBUG, noisy GPU/windowing deps at WARN. Set
//! `RUST_LOG` to override everything (e.g. `RUST_LOG=planet_explorer=trace`).
//!
//! Log location (first match wins):
//!   $PLANET_LOG            — explicit full file path
//!   $PLANET_LOG_DIR/...    — directory, file name fixed
//!   /Users/Shared/planet-explorer/planet-explorer.log  — default (cross-account
//!                            readable: the GUI account writes it, we read it)

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing_subscriber::{fmt, prelude::*, EnvFilter, Layer};

/// Our crate at DEBUG; dependencies that spam per-frame held to WARN.
const DEFAULT_FILTER: &str = "debug,wgpu=warn,wgpu_core=warn,wgpu_hal=warn,naga=warn,winit=warn";
/// stderr (for terminal runs) is quieter than the file.
const STDERR_FILTER: &str = "info,wgpu=warn,wgpu_core=warn,wgpu_hal=warn,naga=warn,winit=warn";

/// Initialise logging. Returns the resolved log file path so the caller can tell
/// the user where to look. Safe to call exactly once at startup.
pub fn init() -> PathBuf {
    let path = log_path();
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
        // World rwx so a different account can create the file inside it.
        set_mode(dir, 0o777);
    }
    // The log file is a diagnostic aid, not a launch dependency: if the path is
    // unwritable (a locked-down or read-only FS — likelier on the Windows target
    // than the dev Mac), degrade to stderr-only with a warning rather than
    // refusing to start. `Option<Layer>` is itself a `Layer` (None = no-op).
    let file_layer = match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(file) => {
            // World rw so either account can append and read the shared log.
            set_mode(&path, 0o666);
            Some(
                fmt::layer()
                    .with_ansi(false)
                    .with_target(true)
                    .with_thread_names(true)
                    .with_writer(Mutex::new(file))
                    .with_filter(env_filter(DEFAULT_FILTER)),
            )
        }
        Err(e) => {
            eprintln!("warning: cannot open log file {}: {e}; logging to stderr only", path.display());
            None
        }
    };

    let stderr_layer = fmt::layer()
        .with_ansi(true)
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_filter(env_filter(STDERR_FILTER));

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stderr_layer)
        .init();

    install_panic_hook();
    path
}

/// Build an `EnvFilter` from `RUST_LOG` if it is set and non-empty, otherwise
/// from `fallback`. An empty `RUST_LOG` is treated as unset rather than
/// "silence everything" — that footgun would quietly disable the whole log.
fn env_filter(fallback: &str) -> EnvFilter {
    match std::env::var("RUST_LOG") {
        Ok(v) if !v.trim().is_empty() => EnvFilter::new(v),
        _ => EnvFilter::new(fallback),
    }
}

fn log_path() -> PathBuf {
    if let Ok(p) = std::env::var("PLANET_LOG") {
        return PathBuf::from(p);
    }
    let dir = std::env::var("PLANET_LOG_DIR").unwrap_or_else(|_| default_log_dir());
    Path::new(&dir).join("planet-explorer.log")
}

/// Default log directory, per platform. On macOS we use the shared folder so a
/// second account can read the log; elsewhere (incl. the planned Windows box) we
/// fall back to the OS temp dir, which is always writable.
fn default_log_dir() -> String {
    #[cfg(target_os = "macos")]
    {
        "/Users/Shared/planet-explorer".to_string()
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::env::temp_dir().join("planet-explorer").to_string_lossy().into_owned()
    }
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    // Best-effort: a non-owner can't chmod, which is fine — the first creator
    // sets the permissive mode and both accounts inherit it.
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) {}

/// Route panics through the log (with location) before the default handler runs,
/// so a crash on the other account is always captured in the file.
fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "<unknown>".to_string());
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("<non-string panic payload>");
        tracing::error!(target: "panic", %location, "PANIC: {msg}");
        default(info);
    }));
}
