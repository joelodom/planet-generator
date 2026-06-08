//! Build script: stamp the git commit and build date into the binary so every
//! log line set can be traced back to the exact source it was built from.
//! Exposed as `env!("GIT_HASH")` and `env!("BUILD_DATE")`.

use std::process::Command;

fn main() {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };

    let mut hash = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    // Mark uncommitted builds so a log from a dirty tree is never mistaken for a
    // clean commit.
    if git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty()) {
        hash.push_str("-dirty");
    }
    println!("cargo:rustc-env=GIT_HASH={hash}");

    // Local build time (the headless dev account and the GUI account that runs the
    // app share one timezone), e.g. "2026-06-08 07:16:59 EDT". Display-only — shown
    // on the ESC overlay and in the startup log; not parsed anywhere.
    let date = Command::new("date")
        .args(["+%Y-%m-%d %H:%M:%S %Z"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=BUILD_DATE={date}");

    // Force this script to re-run on EVERY build so BUILD_DATE/GIT_HASH reflect
    // the actual build, not just the last commit. Referencing a path that never
    // exists makes Cargo treat the script as dirty each build.
    println!("cargo:rerun-if-changed=.always-rerun-build-stamp");
}
