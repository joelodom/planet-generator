#!/usr/bin/env python3
"""Generate the app icon (a macOS-style squircle) from assets/planet.png.

From one rendered 1024px squircle this writes BOTH platform icons so they stay
visually identical:
  * assets/AppIcon.ico   — Windows (multi-resolution; embedded in the .exe by
                           build.rs via the `winresource` crate)
  * assets/AppIcon.icns  — macOS (built with `iconutil`, if available; copied into
                           the .app bundle by package_macos.sh)
Also drops /tmp/appicon_1024.png for inspection. Pure Pillow + iconutil, no numpy.

Run it on the Mac (iconutil is macOS-only); the .ico is produced everywhere.
"""
from PIL import Image, ImageDraw, ImageFilter
import os
import shutil
import subprocess
import tempfile

SIZE = 1024
ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")

# --- deep-space background with a soft central glow ---
# radial_gradient: 0 at centre, 255 at corners -> invert for a bright centre.
glow_mask = Image.radial_gradient("L").resize((SIZE, SIZE)).point(lambda v: int((255 - v) * 0.8))
base = Image.new("RGB", (SIZE, SIZE), (6, 10, 26))
glow = Image.new("RGB", (SIZE, SIZE), (24, 34, 74))
bg = Image.composite(glow, base, glow_mask).convert("RGBA")

# --- planet, circularly cropped and scaled ---
planet = Image.open(os.path.join(ASSETS, "planet.png")).convert("RGBA")
pd = planet.size[0]
disc = Image.new("L", (pd, pd), 0)
ImageDraw.Draw(disc).ellipse((0, 0, pd - 1, pd - 1), fill=255)
planet.putalpha(disc)
ps = int(SIZE * 0.80)
planet = planet.resize((ps, ps), Image.LANCZOS)
off = (SIZE - ps) // 2

# soft drop shadow under the planet
shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
sh = Image.new("L", (ps, ps), 0)
ImageDraw.Draw(sh).ellipse((0, 0, ps - 1, ps - 1), fill=150)
shadow.paste((0, 0, 0, 255), (off, off + int(SIZE * 0.015)), sh)
shadow = shadow.filter(ImageFilter.GaussianBlur(SIZE * 0.025))
bg.alpha_composite(shadow)
bg.alpha_composite(planet, (off, off))

# --- squircle mask with margin (macOS rounded-rect look) ---
margin = int(SIZE * 0.085)
radius = int(SIZE * 0.225)
mask = Image.new("L", (SIZE, SIZE), 0)
ImageDraw.Draw(mask).rounded_rectangle(
    [margin, margin, SIZE - margin, SIZE - margin], radius=radius, fill=255
)
icon = Image.composite(bg, Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0)), mask)

png_path = "/tmp/appicon_1024.png"
icon.save(png_path)
print("wrote", png_path)

# --- Windows .ico (multi-resolution; 256 is the ICO max) ---
ico_path = os.path.join(ASSETS, "AppIcon.ico")
icon.save(ico_path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print("wrote", ico_path)

# --- macOS .icns via iconutil (macOS only) ---
icns_path = os.path.join(ASSETS, "AppIcon.icns")
if shutil.which("iconutil"):
    # Standard iconset: (filename, pixel size) pairs iconutil expects.
    variants = [
        ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512), ("icon_512x512@2x.png", 1024),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "AppIcon.iconset")
        os.makedirs(iconset)
        for name, px in variants:
            icon.resize((px, px), Image.LANCZOS).save(os.path.join(iconset, name))
        subprocess.run(["iconutil", "-c", "icns", "-o", icns_path, iconset], check=True)
    print("wrote", icns_path)
else:
    print("iconutil not found (macOS only) — skipped", icns_path, "(the committed .icns is reused)")
