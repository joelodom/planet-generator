@echo off
REM Convenience wrapper so you don't have to change your system's PowerShell
REM execution policy. Just run:  package_windows.cmd  (optionally: -Run, -DynamicCrt)
REM
REM It invokes the .ps1 with an execution-policy bypass scoped to THIS process only
REM -- nothing about your machine's policy is changed.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0package_windows.ps1" %*
