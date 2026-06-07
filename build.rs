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

    let date = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=BUILD_DATE={date}");

    // Rebuild the stamp when HEAD moves (new commit / checkout).
    println!("cargo:rerun-if-changed=.git/HEAD");
}
