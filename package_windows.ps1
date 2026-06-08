<#
.SYNOPSIS
    Build Planet Explorer as a single, self-contained Windows .exe.

.DESCRIPTION
    Release-builds the crate and drops a standalone planet-explorer.exe in dist\.
    All assets (textures, music, shaders) are embedded in the binary via
    include_bytes!/include_str!, so the .exe is the only file you need — copy it
    anywhere and run it.

    By default the MSVC C runtime is STATICALLY linked (crt-static), so the exe
    runs on a clean Windows box with no Visual C++ Redistributable installed.
    Pass -DynamicCrt to link it dynamically (smaller exe, needs VC++ redist).

.PARAMETER DynamicCrt
    Link the MSVC runtime dynamically instead of statically.

.PARAMETER Run
    Launch the exe after a successful build.

.EXAMPLE
    .\package_windows.ps1
    .\package_windows.ps1 -Run
    .\package_windows.ps1 -DynamicCrt

.NOTES
    "running scripts is disabled on this system"? That's PowerShell's default
    execution policy blocking unsigned local scripts. Three ways around it:
      * Easiest — run the bundled wrapper, which bypasses the policy for just this
        one invocation (no system change):   package_windows.cmd
      * Or invoke PowerShell with a one-off bypass:
            powershell -ExecutionPolicy Bypass -File .\package_windows.ps1
      * Or allow local scripts for your user (persistent):
            Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

    Prerequisites:
      * Rust (MSVC toolchain): https://rustup.rs  — `rustup default stable-msvc`
        Edition 2024 needs Rust 1.85 or newer.
      * Up-to-date NVIDIA driver for the RTX 5090 (DX12). Vulkan also works.
      * git on PATH (optional — only used to stamp the version; build works without).
#>
[CmdletBinding()]
param(
    [switch]$DynamicCrt,
    [switch]$Run
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$BinName = 'planet-explorer'

# Run from the script's own directory so relative paths resolve regardless of CWD.
Set-Location -Path $PSScriptRoot

Write-Host '>> checking toolchain' -ForegroundColor Cyan
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw 'cargo not found on PATH. Install Rust (MSVC toolchain) from https://rustup.rs'
}

# Single source of truth for the version: the crate version in Cargo.toml,
# stamped with the git short hash (mirrors package_macos.sh).
# Version stamp is cosmetic — degrade to a placeholder rather than hard-fail if the
# manifest can't be parsed (matches package_macos.sh's best-effort approach).
$VersionMatch = Select-String -Path 'Cargo.toml' -Pattern '^\s*version\s*=\s*"(.*)"' | Select-Object -First 1
$Version = if ($VersionMatch) { $VersionMatch.Matches[0].Groups[1].Value } else { 'unknown' }
$GitHash = 'unknown'
if (Get-Command git -ErrorAction SilentlyContinue) {
    try { $GitHash = (git rev-parse --short HEAD).Trim() } catch { $GitHash = 'unknown' }
}
Write-Host "   planet-explorer $Version+$GitHash" -ForegroundColor DarkGray

# Static CRT => the .exe needs no VC++ Redistributable on the target machine.
# This is what makes it a genuine "single file you can copy and run".
if ($DynamicCrt) {
    Write-Host '>> linking MSVC runtime dynamically (needs VC++ redist on target)' -ForegroundColor Yellow
    Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
} else {
    Write-Host '>> static MSVC runtime (standalone exe, no redist needed)' -ForegroundColor Cyan
    $env:RUSTFLAGS = '-C target-feature=+crt-static'
}

Write-Host '>> building release binary (this can take a few minutes the first time)' -ForegroundColor Cyan
cargo build --release
if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }

$SrcExe = Join-Path 'target\release' "$BinName.exe"
if (-not (Test-Path $SrcExe)) { throw "expected binary not found: $SrcExe" }

# Stage the standalone exe in dist\ with the version baked into the filename so
# you can tell builds apart at a glance.
$DistDir = 'dist'
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
$Stamped = Join-Path $DistDir "$BinName-$Version+$GitHash.exe"
$Plain   = Join-Path $DistDir "$BinName.exe"
Copy-Item $SrcExe $Stamped -Force
Copy-Item $SrcExe $Plain   -Force

$SizeMB = [math]::Round((Get-Item $Plain).Length / 1MB, 1)

Write-Host ''
Write-Host 'Done.' -ForegroundColor Green
Write-Host "  Exe     : $((Resolve-Path $Plain).Path)  ($SizeMB MB, self-contained)"
Write-Host "  Stamped : $((Resolve-Path $Stamped).Path)"
Write-Host ''
Write-Host 'Run it:           ' -NoNewline; Write-Host ".\$Plain"
Write-Host 'Specific seed:    ' -NoNewline; Write-Host ".\$Plain --seed 12345"
Write-Host 'Confirm build:    ' -NoNewline; Write-Host ".\$Plain --version"
Write-Host 'US units in HUD:  ' -NoNewline; Write-Host ".\$Plain --units us"
Write-Host ''
Write-Host "Log file: %TEMP%\planet-explorer\planet-explorer.log  (override with `$env:PLANET_LOG)" -ForegroundColor DarkGray
Write-Host "Which GPU got picked: look for the 'gpu adapter selected' line in that log." -ForegroundColor DarkGray

if ($Run) {
    Write-Host ''
    Write-Host '>> launching' -ForegroundColor Cyan
    & (Resolve-Path $Plain).Path
}
