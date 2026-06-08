#!/usr/bin/env bash
# Build the release binary and assemble a double-clickable macOS .app bundle.
#
#   ./package_macos.sh
#
# Output: dist/Planet Explorer.app  (and a copy in /Users/Shared for cross-account access)
set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="Planet Explorer"
BIN_NAME="planet-explorer"
BUNDLE_ID="com.joelodom.planet-explorer"
# Single source of truth: the crate version in Cargo.toml, plus the git commit.
VERSION="$(sed -n 's/^version *= *"\(.*\)".*/\1/p' Cargo.toml | head -1)"
GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BUILD_VERSION="${VERSION}+${GIT_HASH}"

echo ">> building release binary"
cargo build --release

APP_DIR="dist/${APP_NAME}.app"
MACOS_DIR="${APP_DIR}/Contents/MacOS"
RES_DIR="${APP_DIR}/Contents/Resources"

echo ">> assembling ${APP_DIR}"
rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}" "${RES_DIR}"
cp "target/release/${BIN_NAME}" "${MACOS_DIR}/${BIN_NAME}"
chmod 755 "${MACOS_DIR}/${BIN_NAME}"

# App icon (regenerate from planet.png with ./make_icon.py, which emits both this
# .icns and the Windows assets/AppIcon.ico). The Windows .exe embeds the .ico itself
# via build.rs; here we just copy the .icns into the bundle.
ICON_LINE=""
if [ -f assets/AppIcon.icns ]; then
    cp assets/AppIcon.icns "${RES_DIR}/AppIcon.icns"
    ICON_LINE="    <key>CFBundleIconFile</key>
    <string>AppIcon</string>"
fi

cat > "${APP_DIR}/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleExecutable</key>
    <string>${BIN_NAME}</string>
${ICON_LINE}
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleVersion</key>
    <string>${BUILD_VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.games</string>
</dict>
</plist>
PLIST

# Ad-hoc code signature so Gatekeeper doesn't flag it as "damaged" when copied
# between accounts. (No Apple Developer ID needed for local use.)
if command -v codesign >/dev/null 2>&1; then
    echo ">> ad-hoc signing"
    codesign --force --deep --sign - "${APP_DIR}" || echo "   (codesign skipped)"
fi

# Make everything world-readable/executable so a different macOS account can run it.
chmod -R a+rX "${APP_DIR}"

# Drop a copy in the shared folder for easy cross-account copying.
SHARED="/Users/Shared/${APP_NAME}.app"
if [ -w /Users/Shared ]; then
    echo ">> copying to ${SHARED}"
    rm -rf "${SHARED}"
    cp -R "${APP_DIR}" "${SHARED}"
    chmod -R a+rX "${SHARED}"
fi

echo ""
echo "Done."
echo "  Bundle : ${PWD}/${APP_DIR}"
[ -d "${SHARED}" ] && echo "  Shared : ${SHARED}"
echo ""
echo "Run it:        open \"${PWD}/${APP_DIR}\""
echo "Specific seed: \"${PWD}/${APP_DIR}/Contents/MacOS/${BIN_NAME}\" --seed 12345"
