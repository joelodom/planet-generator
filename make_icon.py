#!/usr/bin/env python3
"""Generate a macOS-style app icon (squircle) from assets/planet.png.

Outputs /tmp/appicon_1024.png; package_macos.sh turns it into AppIcon.icns.
Pure Pillow, no numpy.
"""
from PIL import Image, ImageDraw, ImageFilter
import os

SIZE = 1024
ROOT = os.path.dirname(os.path.abspath(__file__))

# --- deep-space background with a soft central glow ---
# radial_gradient: 0 at centre, 255 at corners -> invert for a bright centre.
glow_mask = Image.radial_gradient("L").resize((SIZE, SIZE)).point(lambda v: int((255 - v) * 0.8))
base = Image.new("RGB", (SIZE, SIZE), (6, 10, 26))
glow = Image.new("RGB", (SIZE, SIZE), (24, 34, 74))
bg = Image.composite(glow, base, glow_mask).convert("RGBA")

# --- planet, circularly cropped and scaled ---
planet = Image.open(os.path.join(ROOT, "assets/planet.png")).convert("RGBA")
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

out = "/tmp/appicon_1024.png"
icon.save(out)
print("wrote", out)
