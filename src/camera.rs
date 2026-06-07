//! Google-Earth-style camera: it orbits a *focus point* on the planet surface.
//!
//! Instead of flying a free first-person eye (which is disorienting on a sphere),
//! the camera always looks at a focus point on the ground and is parameterised
//! the way Google Earth is:
//!   - `focus`    — the lat/long on the surface being looked at
//!   - `distance` — how far the eye is from that point (zoom)
//!   - `heading`  — compass rotation around the focus
//!   - `tilt`     — angle from straight-down (0) toward the horizon
//!
//! Everything is keyboard driven and scale-aware (panning/zooming speed grows
//! with distance), so it feels the same from orbit down to street level.

use crate::planet::{self, Planet, HEIGHT_SCALE, PLANET_RADIUS};
use glam::{Mat4, Quat, Vec3};

// Distances are in render units (10 m each). Min ~15 m off the deck, max well
// outside the planet so you can see the full globe.
const MIN_DIST: f32 = 1.5;
const MAX_DIST: f32 = 3.0 * PLANET_RADIUS;
const MAX_TILT: f32 = 1.30; // ~74.5°, keeps the eye comfortably above the ground

const ZOOM_RATE: f32 = 1.6; // e-folds per second
const ROT_RATE: f32 = 1.3; // rad/s
const TILT_RATE: f32 = 1.2; // rad/s

#[derive(Default)]
struct Keys {
    pan_fwd: bool,
    pan_back: bool,
    pan_left: bool,
    pan_right: bool,
    zoom_in: bool,
    zoom_out: bool,
    rot_left: bool,
    rot_right: bool,
    tilt_more: bool, // toward the horizon
    tilt_less: bool, // toward top-down
    boost: bool,
}

pub struct Camera {
    /// Unit direction to the surface point under the camera's focus.
    pub focus: Vec3,
    /// Eye distance from the focus point.
    distance: f32,
    /// Compass heading (radians).
    heading: f32,
    /// Tilt from straight-down (0) toward the horizon.
    tilt: f32,

    pub aspect: f32,
    pub fov_y: f32,
    keys: Keys,
}

impl Camera {
    pub fn new(_planet: &Planet, focus: Vec3) -> Self {
        Self {
            focus: focus.normalize(),
            distance: PLANET_RADIUS * 2.2, // start with a full-globe view
            heading: 0.0,
            tilt: 0.0,
            aspect: 1.0,
            fov_y: 60f32.to_radians(),
            keys: Keys::default(),
        }
    }

    // --- local frame & eye --------------------------------------------------

    /// North/east tangents at the focus (north = toward +Y pole, projected).
    fn frame(&self) -> (Vec3, Vec3) {
        let up = self.focus;
        let mut north = Vec3::Y - up * Vec3::Y.dot(up);
        if north.length_squared() < 1e-5 {
            north = planet::tangent_basis(up).0; // at a pole, any tangent
        }
        north = north.normalize();
        let east = up.cross(north).normalize();
        (north, east)
    }

    /// Horizontal look direction (tangent) given the current heading.
    fn look_h(&self) -> Vec3 {
        let (north, east) = self.frame();
        (north * self.heading.cos() + east * self.heading.sin()).normalize()
    }

    /// Eye position, look direction, and view-up — the camera basis.
    fn view(&self, planet: &Planet) -> (Vec3, Vec3, Vec3) {
        let up = self.focus;
        let surface_r = planet.surface_radius(self.focus);
        let focus_point = self.focus * surface_r;
        let look_h = self.look_h();
        let (st, ct) = (self.tilt.sin(), self.tilt.cos());

        // Eye sits above-and-behind the focus; raising tilt swings it down toward
        // the horizon.
        let mut eye = focus_point + up * (self.distance * ct) - look_h * (self.distance * st);

        // Never let the eye dip below the surface (e.g. steep tilt near ground).
        let eye_dir = eye.normalize();
        let min_r = planet.surface_radius(eye_dir) + 1.0;
        if eye.length() < min_r {
            eye = eye_dir * min_r;
        }

        let look_dir = (focus_point - eye).normalize();
        // View-up = focus normal with the look component removed; degenerate only
        // at exact top-down, where screen-up is the heading direction.
        let mut up_vec = up - look_dir * up.dot(look_dir);
        if up_vec.length_squared() < 1e-6 {
            up_vec = look_h;
        }
        (eye, look_dir, up_vec.normalize())
    }

    pub fn position(&self, planet: &Planet) -> Vec3 {
        self.view(planet).0
    }

    /// Height of the eye above the terrain directly beneath it.
    pub fn altitude(&self) -> f32 {
        // Cheap proxy used for fog/near-far; exact terrain height not needed.
        self.distance
    }

    // --- input --------------------------------------------------------------

    pub fn set_aspect(&mut self, w: u32, h: u32) {
        self.aspect = (w.max(1) as f32) / (h.max(1) as f32);
    }

    pub fn key(&mut self, code: KeyAction, pressed: bool) {
        match code {
            KeyAction::PanForward => self.keys.pan_fwd = pressed,
            KeyAction::PanBack => self.keys.pan_back = pressed,
            KeyAction::PanLeft => self.keys.pan_left = pressed,
            KeyAction::PanRight => self.keys.pan_right = pressed,
            KeyAction::ZoomIn => self.keys.zoom_in = pressed,
            KeyAction::ZoomOut => self.keys.zoom_out = pressed,
            KeyAction::RotateLeft => self.keys.rot_left = pressed,
            KeyAction::RotateRight => self.keys.rot_right = pressed,
            KeyAction::TiltMore => self.keys.tilt_more = pressed,
            KeyAction::TiltLess => self.keys.tilt_less = pressed,
            KeyAction::Boost => self.keys.boost = pressed,
        }
    }

    /// Drop the focus onto a random surface point, zoomed in at a nice angle.
    pub fn teleport(&mut self, _planet: &Planet, dir: Vec3) {
        self.focus = dir.normalize();
        self.distance = 12.0;
        self.tilt = 0.85;
    }

    // --- per-frame update ---------------------------------------------------

    pub fn update(&mut self, dt: f32, planet: &Planet) {
        let boost = if self.keys.boost { 4.0 } else { 1.0 };

        // Zoom (multiplicative, so it's smooth across scales).
        let zoom = (self.keys.zoom_out as i32 - self.keys.zoom_in as i32) as f32;
        if zoom != 0.0 {
            self.distance = (self.distance * (ZOOM_RATE * zoom * boost * dt).exp()).clamp(MIN_DIST, MAX_DIST);
        }

        // Rotate (heading) and tilt.
        self.heading += (self.keys.rot_right as i32 - self.keys.rot_left as i32) as f32 * ROT_RATE * boost * dt;
        self.tilt = (self.tilt + (self.keys.tilt_more as i32 - self.keys.tilt_less as i32) as f32 * TILT_RATE * dt)
            .clamp(0.0, MAX_TILT);

        // Pan the focus across the surface, in screen-forward / screen-right.
        let fwd = self.keys.pan_fwd as i32 - self.keys.pan_back as i32;
        let strafe = self.keys.pan_right as i32 - self.keys.pan_left as i32;
        if fwd != 0 || strafe != 0 {
            let up = self.focus;
            let look_h = self.look_h();
            let right = look_h.cross(up).normalize(); // screen-right tangent
            let mut dir = look_h * fwd as f32 + right * strafe as f32;
            if dir.length_squared() > 1e-6 {
                dir = dir.normalize();
                // Pan speed grows with zoom but is capped so far-out panning is sane.
                let pan_world = (self.distance * 0.6).clamp(3.0, PLANET_RADIUS * 0.4) * boost;
                let ang = pan_world / PLANET_RADIUS * dt;
                let axis = self.focus.cross(dir).normalize_or_zero();
                if axis != Vec3::ZERO {
                    self.focus = (Quat::from_axis_angle(axis, ang) * self.focus).normalize();
                }
            }
        }
        let _ = planet;
    }

    // --- matrices -----------------------------------------------------------

    /// Near/far planes tuned per frame from the eye's horizon. On an Earth-sized
    /// planet a fixed far plane would either clip the globe or destroy depth
    /// precision near the ground, so far tracks the visible horizon (plus the
    /// distance mountains can poke above it) and near tracks the focus distance.
    fn near_far(&self, eye: Vec3) -> (f32, f32) {
        let r = PLANET_RADIUS;
        let eye_r = eye.length();
        let horizon = if eye_r > r { (eye_r * eye_r - r * r).max(0.0).sqrt() } else { 0.0 };
        // Peaks beyond the geometric horizon are still visible.
        let mtn = (2.0 * r * HEIGHT_SCALE * 1.4).sqrt();
        let far = horizon + mtn + self.distance + r * 0.01 + 100.0;
        // Nothing is closer than ~the focus point, which is `distance` away.
        let near = (self.distance * 0.25).clamp(0.1, far * 0.4);
        (near, far)
    }

    pub fn view_proj(&self, planet: &Planet) -> (Mat4, Mat4, Vec3) {
        let (eye, look_dir, up_vec) = self.view(planet);
        let view = Mat4::look_to_rh(eye, look_dir, up_vec);
        let (near, far) = self.near_far(eye);
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect, near, far);
        (proj * view, view, eye)
    }

    /// Fog thickens near the ground (hides far LOD pop-in) and vanishes from orbit.
    pub fn fog_density(&self) -> f32 {
        let t = (1.0 - self.distance / 3000.0).clamp(0.0, 1.0);
        t * t * (1.0 / 4000.0)
    }

    pub fn lat_lon(&self) -> (f32, f32) {
        let lat = self.focus.y.clamp(-1.0, 1.0).asin().to_degrees();
        let lon = self.focus.z.atan2(self.focus.x).to_degrees();
        (lat, lon)
    }
}

/// Navigation intents, decoupled from physical key codes (mapped in `main`).
#[derive(Clone, Copy)]
pub enum KeyAction {
    PanForward,
    PanBack,
    PanLeft,
    PanRight,
    ZoomIn,
    ZoomOut,
    RotateLeft,
    RotateRight,
    TiltMore,
    TiltLess,
    Boost,
}
