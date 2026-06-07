//! The camera: a single continuum from orbit to ground, with no hard mode
//! switches.
//!
//! The trick is the state representation. Instead of tracking a free-floating
//! eye, the camera stores an `anchor` (a unit direction = the lat/long it's
//! above) and an `altitude` (height above the surface along that direction).
//! Position is always `anchor * (surface_radius + altitude)`. Because both orbit
//! and surface controls only ever nudge these two quantities, the camera can
//! never end up underground or off the planet, and zooming from space to a
//! hilltop is perfectly continuous — the controls simply reinterpret the same
//! state as you descend.

use crate::planet::{self, Planet};
use glam::{Mat4, Quat, Vec3};
use std::f32::consts::PI;

/// Altitude (world units) above which orbit-style controls take over.
const ORBIT_ALT: f32 = 320.0;
/// Never let the eye get closer than this to the surface.
const MIN_ALT: f32 = 1.6;
const MAX_ALT: f32 = 9000.0;

#[derive(Default)]
struct Keys {
    fwd: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    sprint: bool,
}

pub struct Camera {
    pub anchor: Vec3,
    pub altitude: f32,
    pub look_dir: Vec3,
    pub aspect: f32,
    pub fov_y: f32,

    keys: Keys,
    pub left_mouse: bool,
    pub right_mouse: bool,
    pub free_look: bool,
    speed_mult: f32,
}

impl Camera {
    pub fn new(planet: &Planet, anchor: Vec3) -> Self {
        let anchor = anchor.normalize();
        let mut cam = Self {
            anchor,
            altitude: 2600.0,
            look_dir: -anchor,
            aspect: 1.0,
            fov_y: 60f32.to_radians(),
            keys: Keys::default(),
            left_mouse: false,
            right_mouse: false,
            free_look: false,
            speed_mult: 1.0,
        };
        cam.clamp_to_surface(planet);
        cam
    }

    fn orbit_mode(&self) -> bool {
        self.altitude > ORBIT_ALT
    }

    /// Local up at the current position (away from planet centre).
    fn up(&self) -> Vec3 {
        self.anchor
    }

    pub fn position(&self, planet: &Planet) -> Vec3 {
        self.anchor * (planet.surface_radius(self.anchor) + self.altitude)
    }

    pub fn altitude(&self) -> f32 {
        self.altitude
    }

    // --- input -------------------------------------------------------------

    pub fn set_aspect(&mut self, w: u32, h: u32) {
        self.aspect = (w.max(1) as f32) / (h.max(1) as f32);
    }

    pub fn key(&mut self, code: KeyAction, pressed: bool) {
        match code {
            KeyAction::Forward => self.keys.fwd = pressed,
            KeyAction::Back => self.keys.back = pressed,
            KeyAction::Left => self.keys.left = pressed,
            KeyAction::Right => self.keys.right = pressed,
            KeyAction::Ascend => self.keys.up = pressed,
            KeyAction::Descend => self.keys.down = pressed,
            KeyAction::Sprint => self.keys.sprint = pressed,
        }
    }

    pub fn adjust_speed(&mut self, factor: f32) {
        self.speed_mult = (self.speed_mult * factor).clamp(0.1, 40.0);
    }

    pub fn toggle_free_look(&mut self) {
        self.free_look = !self.free_look;
    }

    /// Mouse motion. Routed to orbiting or free-look depending on mode/buttons.
    pub fn mouse_motion(&mut self, dx: f32, dy: f32) {
        let orbit = self.orbit_mode();
        if self.left_mouse && orbit {
            self.orbit_drag(dx, dy);
        } else if self.right_mouse || self.free_look || (self.left_mouse && !orbit) {
            self.mouse_look(dx, dy);
        }
    }

    fn orbit_drag(&mut self, dx: f32, dy: f32) {
        let k = 0.005;
        // Yaw around world up, pitch around the current tangent.
        let yaw = Quat::from_rotation_y(-dx * k);
        let right = self.anchor.cross(Vec3::Y).normalize_or_zero();
        let right = if right == Vec3::ZERO { Vec3::X } else { right };
        let pitch = Quat::from_axis_angle(right, -dy * k);
        self.anchor = (pitch * yaw * self.anchor).normalize();
        self.look_dir = -self.anchor; // keep the globe centred while orbiting
    }

    fn mouse_look(&mut self, dx: f32, dy: f32) {
        let k = 0.0032;
        let up = self.up();
        let right = self.look_dir.cross(up).normalize_or_zero();
        let right = if right == Vec3::ZERO { planet::tangent_basis(up).0 } else { right };
        let yaw = Quat::from_axis_angle(up, -dx * k);
        let pitch = Quat::from_axis_angle(right, -dy * k);
        let mut dir = (yaw * pitch * self.look_dir).normalize();
        // Clamp so we never look exactly along local up/down (avoids roll flips).
        let cos = dir.dot(up).clamp(-1.0, 1.0);
        let ang = cos.acos();
        let limit = 0.06;
        if ang < limit {
            dir = (dir - up * (cos - limit.cos())).normalize();
        } else if ang > PI - limit {
            dir = (dir - up * (cos + limit.cos())).normalize();
        }
        self.look_dir = dir;
    }

    pub fn scroll(&mut self, delta: f32) {
        // Multiplicative zoom toward/away from the surface: slows as you near
        // the ground, so the descent into surface mode is smooth.
        let factor = (1.0 - delta * 0.12).clamp(0.5, 2.0);
        self.altitude = (self.altitude * factor).clamp(MIN_ALT, MAX_ALT);
    }

    /// Drop the camera onto a random point on the surface, looking at the horizon.
    pub fn teleport(&mut self, planet: &Planet, dir: Vec3) {
        self.anchor = dir.normalize();
        self.altitude = 3.0;
        // Look along a horizontal tangent, tilted slightly down toward the ground.
        let (t, _) = planet::tangent_basis(self.anchor);
        self.look_dir = (t - self.anchor * 0.12).normalize();
        self.clamp_to_surface(planet);
    }

    // --- per-frame update --------------------------------------------------

    pub fn update(&mut self, dt: f32, planet: &Planet) {
        let up = self.up();
        // Horizontal movement basis from where we're looking.
        let mut fwd = self.look_dir - up * self.look_dir.dot(up);
        if fwd.length_squared() < 1e-6 {
            fwd = planet::tangent_basis(up).0;
        }
        fwd = fwd.normalize();
        let right = fwd.cross(up).normalize();

        let mut move_dir = Vec3::ZERO;
        if self.keys.fwd { move_dir += fwd; }
        if self.keys.back { move_dir -= fwd; }
        if self.keys.right { move_dir += right; }
        if self.keys.left { move_dir -= right; }

        let sprint = if self.keys.sprint { 6.0 } else { 1.0 };
        // Speed scales with altitude so the world feels consistent from orbit to ground.
        let base = (self.altitude * 0.55 + 10.0) * self.speed_mult * sprint;

        if move_dir.length_squared() > 1e-6 {
            let tangent = move_dir.normalize();
            let surface_r = planet.surface_radius(self.anchor) + self.altitude;
            let ang = (base * dt / surface_r).min(0.3);
            let axis = self.anchor.cross(tangent).normalize_or_zero();
            if axis != Vec3::ZERO {
                let rot = Quat::from_axis_angle(axis, ang);
                self.anchor = (rot * self.anchor).normalize();
                // Carry the view with us so the horizon stays put as we walk.
                self.look_dir = (rot * self.look_dir).normalize();
            }
        }

        // Vertical: space ascends, descend key lowers. Scales with altitude too.
        let vert = (self.altitude * 0.5 + 8.0) * self.speed_mult * sprint;
        if self.keys.up { self.altitude += vert * dt; }
        if self.keys.down { self.altitude -= vert * dt; }
        self.clamp_to_surface(planet);
    }

    fn clamp_to_surface(&mut self, _planet: &Planet) {
        self.altitude = self.altitude.clamp(MIN_ALT, MAX_ALT);
        self.anchor = self.anchor.normalize();
        self.look_dir = self.look_dir.normalize_or_zero();
        if self.look_dir == Vec3::ZERO {
            self.look_dir = -self.anchor;
        }
    }

    // --- matrices ----------------------------------------------------------

    pub fn near_far(&self) -> (f32, f32) {
        let near = (self.altitude * 0.22).clamp(0.05, 120.0);
        let far = self.altitude + planet::PLANET_RADIUS * 2.2 + 4000.0;
        (near, far)
    }

    pub fn view_proj(&self, planet: &Planet) -> (Mat4, Mat4, Vec3) {
        let pos = self.position(planet);
        let mut up = self.up();
        // If we're looking nearly straight up/down, swap to a tangent up vector.
        if self.look_dir.dot(up).abs() > 0.98 {
            up = planet::tangent_basis(self.anchor).0;
        }
        let view = Mat4::look_to_rh(pos, self.look_dir, up);
        let (near, far) = self.near_far();
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect, near, far);
        (proj * view, view, pos)
    }

    /// Fog thickens near the ground (to hide LOD pop-in) and vanishes in space.
    pub fn fog_density(&self) -> f32 {
        // 0 at high altitude, ~1/450 near the surface.
        let t = (1.0 - (self.altitude / 600.0)).clamp(0.0, 1.0);
        t * t * (1.0 / 450.0)
    }

    pub fn lat_lon(&self) -> (f32, f32) {
        let lat = self.anchor.y.clamp(-1.0, 1.0).asin().to_degrees();
        let lon = self.anchor.z.atan2(self.anchor.x).to_degrees();
        (lat, lon)
    }
}

/// Movement intents, decoupled from physical key codes (set in `main`).
#[derive(Clone, Copy)]
pub enum KeyAction {
    Forward,
    Back,
    Left,
    Right,
    Ascend,
    Descend,
    Sprint,
}
