//! Guided "tour" — a relaxing, screensaver-style autopilot for the camera.
//!
//! It loops three eased phases:
//!   1. **Travel**  — zoom out and glide along a great circle to a new biome.
//!   2. **Descend** — ease down toward the surface at an oblique, scenic angle.
//!   3. **Cruise**  — drift slowly low over the terrain, gently looking around.
//!
//! Everything is smoothed (no snapping) and slow, so it feels calm. The tour only
//! drives the four camera parameters (focus / distance / heading / tilt), so it
//! composes with all the normal rendering and streaming.

use crate::camera::Camera;
use crate::planet::{self, Planet, PLANET_RADIUS, SHORE};
use glam::{Quat, Vec3};
use std::f32::consts::{PI, TAU};

/// Distance (render units) used while gliding between biomes — high enough to see
/// the planet curve and travel fast, low enough to feel like flying, not orbit.
const TRAVEL_DISTANCE_FACTOR: f32 = 0.16; // * PLANET_RADIUS
const TRAVEL_DIST: f32 = PLANET_RADIUS * TRAVEL_DISTANCE_FACTOR;
const TRAVEL_TILT: f32 = 0.55;

// Travel timing: duration ≈ arc / speed, clamped to a calm range.
const TRAVEL_ANGULAR_SPEED: f32 = 0.08; // rad/s
const TRAVEL_DUR_MIN: f32 = 10.0;
const TRAVEL_DUR_MAX: f32 = 22.0;

const DESCEND_DUR: f32 = 9.0;

// Cruise (low flyover): randomized per leg so it never feels mechanical.
const CRUISE_DIST_MIN: f32 = 1800.0; // render units; high enough to clear peaks
const CRUISE_DIST_MAX: f32 = 4500.0;
const CRUISE_TILT_MIN: f32 = 0.85; // oblique, scenic horizon angles (rad)
const CRUISE_TILT_MAX: f32 = 1.18;
const CRUISE_DUR_MIN: f32 = 24.0; // seconds spent exploring a biome
const CRUISE_DUR_MAX: f32 = 38.0;
const CRUISE_PAN_MIN: f32 = 0.000_12; // rad/s great-circle focus drift
const CRUISE_PAN_MAX: f32 = 0.000_32;
const CRUISE_HEAD_MIN: f32 = 0.006; // rad/s gentle look-around
const CRUISE_HEAD_MAX: f32 = 0.018;
const CRUISE_TILT_AMP_MIN: f32 = 0.04; // gentle tilt bob (rad)
const CRUISE_TILT_AMP_MAX: f32 = 0.10;
const CRUISE_TILT_FREQ_MIN: f32 = 0.04; // rad/s
const CRUISE_TILT_FREQ_MAX: f32 = 0.09;
const CRUISE_TILT_CLAMP: f32 = 1.28; // keep below the camera's MAX_TILT
const ARRIVE_HEADING_JITTER: f32 = 0.6; // ± rad turn as we settle onto a biome

// Destination selection.
const DEST_ARC_MIN: f32 = 0.6; // rad between consecutive biomes (not too near/far)
const DEST_ARC_MAX: f32 = 2.1;
const DEST_MIN_HEIGHT: f32 = 5.0; // prefer solid land, above the coast (render units)

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Travel,
    Descend,
    Cruise,
}

pub struct Tour {
    phase: Phase,
    t: f32,
    dur: f32,

    // Eased segment endpoints (Travel / Descend).
    s_focus: Vec3,
    s_dist: f32,
    s_head: f32,
    s_tilt: f32,
    e_focus: Vec3,
    e_dist: f32,
    e_head: f32,
    e_tilt: f32,

    // Cruise drift state.
    c_focus: Vec3,
    c_dist: f32,
    c_head: f32,
    drift_axis: Vec3,
    pan_rate: f32,
    head_rate: f32,
    base_tilt: f32,
    tilt_amp: f32,
    tilt_freq: f32,
}

impl Tour {
    /// Begin a tour from the camera's current view.
    pub fn new(cam: &Camera, planet: &Planet) -> Self {
        let mut tour = Self {
            phase: Phase::Travel,
            t: 0.0,
            dur: 1.0,
            s_focus: cam.focus,
            s_dist: cam.distance(),
            s_head: cam.heading(),
            s_tilt: cam.tilt(),
            e_focus: cam.focus,
            e_dist: cam.distance(),
            e_head: cam.heading(),
            e_tilt: cam.tilt(),
            c_focus: cam.focus,
            c_dist: cam.distance(),
            c_head: cam.heading(),
            drift_axis: Vec3::Y,
            pan_rate: 0.0,
            head_rate: 0.0,
            base_tilt: cam.tilt(),
            tilt_amp: 0.0,
            tilt_freq: 0.0,
        };
        tour.begin_travel(cam.focus, cam.distance(), cam.heading(), cam.tilt(), planet);
        tour
    }

    pub fn update(&mut self, dt: f32, planet: &Planet, cam: &mut Camera) {
        self.t += dt;
        match self.phase {
            Phase::Travel | Phase::Descend => {
                let s = planet::smoothstep(0.0, 1.0, (self.t / self.dur).min(1.0));
                let focus = slerp_dir(self.s_focus, self.e_focus, s);
                // Exponential (multiplicative) zoom feels natural across scales.
                let dist = self.s_dist * (self.e_dist / self.s_dist).powf(s);
                let head = self.s_head + shortest(self.e_head - self.s_head) * s;
                let tilt = self.s_tilt + (self.e_tilt - self.s_tilt) * s;
                cam.set_view(focus, dist, head, tilt);

                if self.t >= self.dur {
                    if self.phase == Phase::Travel {
                        self.begin_descend(planet);
                    } else {
                        self.begin_cruise();
                    }
                }
            }
            Phase::Cruise => {
                // Drift the focus slowly along a great circle and pan the view.
                self.c_focus = (Quat::from_axis_angle(self.drift_axis, self.pan_rate * dt) * self.c_focus).normalize();
                self.c_head += self.head_rate * dt;
                let tilt = (self.base_tilt + (self.t * self.tilt_freq).sin() * self.tilt_amp).clamp(0.0, CRUISE_TILT_CLAMP);
                cam.set_view(self.c_focus, self.c_dist, self.c_head, tilt);

                if self.t >= self.dur {
                    self.begin_travel(self.c_focus, self.c_dist, self.c_head, self.base_tilt, planet);
                }
            }
        }
    }

    fn begin_travel(&mut self, from_focus: Vec3, from_dist: f32, from_head: f32, from_tilt: f32, planet: &Planet) {
        let dest = pick_destination(from_focus, planet);
        let arc = from_focus.angle_between(dest);

        // Heading that roughly faces the direction of travel.
        let (north, east) = frame(from_focus);
        let tangent = dest - from_focus * from_focus.dot(dest);
        let bearing = if tangent.length_squared() > 1e-9 {
            let t = tangent.normalize();
            t.dot(east).atan2(t.dot(north))
        } else {
            from_head
        };

        self.phase = Phase::Travel;
        self.t = 0.0;
        self.dur = (arc / TRAVEL_ANGULAR_SPEED).clamp(TRAVEL_DUR_MIN, TRAVEL_DUR_MAX);
        self.set_segment(from_focus, from_dist, from_head, from_tilt, dest, TRAVEL_DIST, bearing, TRAVEL_TILT);
    }

    fn begin_descend(&mut self, _planet: &Planet) {
        // High enough to clear most peaks but still a flying view; the camera's
        // ground guard gently lifts it over the rare tall summit.
        let cruise_dist = rand_range(CRUISE_DIST_MIN, CRUISE_DIST_MAX);
        let cruise_tilt = rand_range(CRUISE_TILT_MIN, CRUISE_TILT_MAX);
        let arrive_head = self.e_head + rand_range(-ARRIVE_HEADING_JITTER, ARRIVE_HEADING_JITTER);
        self.phase = Phase::Descend;
        self.t = 0.0;
        self.dur = DESCEND_DUR;
        // Start from where Travel ended; settle onto the biome.
        self.set_segment(self.e_focus, self.e_dist, self.e_head, self.e_tilt, self.e_focus, cruise_dist, arrive_head, cruise_tilt);
    }

    fn begin_cruise(&mut self) {
        self.phase = Phase::Cruise;
        self.t = 0.0;
        self.dur = rand_range(CRUISE_DUR_MIN, CRUISE_DUR_MAX);
        self.c_focus = self.e_focus;
        self.c_dist = self.e_dist;
        self.c_head = self.e_head;

        // Drift in a random direction along the surface.
        let (north, east) = frame(self.c_focus);
        let a = rand_range(0.0, TAU);
        let dir = north * a.cos() + east * a.sin();
        self.drift_axis = self.c_focus.cross(dir).normalize_or_zero();
        if self.drift_axis == Vec3::ZERO {
            self.drift_axis = Vec3::Y;
        }
        self.pan_rate = rand_range(CRUISE_PAN_MIN, CRUISE_PAN_MAX) * rand_sign();
        self.head_rate = rand_range(CRUISE_HEAD_MIN, CRUISE_HEAD_MAX) * rand_sign();
        self.base_tilt = self.e_tilt;
        self.tilt_amp = rand_range(CRUISE_TILT_AMP_MIN, CRUISE_TILT_AMP_MAX);
        self.tilt_freq = rand_range(CRUISE_TILT_FREQ_MIN, CRUISE_TILT_FREQ_MAX);
    }

    #[allow(clippy::too_many_arguments)]
    fn set_segment(&mut self, sf: Vec3, sd: f32, sh: f32, st: f32, ef: Vec3, ed: f32, eh: f32, et: f32) {
        self.s_focus = sf;
        self.s_dist = sd;
        self.s_head = sh;
        self.s_tilt = st;
        self.e_focus = ef;
        self.e_dist = ed;
        self.e_head = eh;
        self.e_tilt = et;
    }
}

/// North/east tangent basis at a surface point (north = toward +Y, projected).
fn frame(up: Vec3) -> (Vec3, Vec3) {
    let mut north = Vec3::Y - up * Vec3::Y.dot(up);
    if north.length_squared() < 1e-5 {
        north = planet::tangent_basis(up).0;
    }
    north = north.normalize();
    let east = up.cross(north).normalize();
    (north, east)
}

/// Pick a varied land destination a comfortable distance from `current`.
fn pick_destination(current: Vec3, planet: &Planet) -> Vec3 {
    for _ in 0..400 {
        let d = random_unit();
        let arc = current.angle_between(d);
        if !(DEST_ARC_MIN..DEST_ARC_MAX).contains(&arc) {
            continue;
        }
        if planet.height(d) > DEST_MIN_HEIGHT {
            return d; // solid land, above the coast
        }
    }
    // Fallback: any land at all.
    for _ in 0..400 {
        let d = random_unit();
        if planet.height(d) > SHORE {
            return d;
        }
    }
    random_unit()
}

/// Spherical interpolation between two unit directions.
fn slerp_dir(a: Vec3, b: Vec3, s: f32) -> Vec3 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    let ang = dot.acos();
    if ang < 1e-4 {
        return b;
    }
    let axis = a.cross(b);
    if axis.length_squared() < 1e-9 {
        return b; // (anti)parallel
    }
    (Quat::from_axis_angle(axis.normalize(), ang * s) * a).normalize()
}

/// Shortest signed equivalent of an angle delta, in (-π, π].
fn shortest(delta: f32) -> f32 {
    (delta + PI).rem_euclid(TAU) - PI
}

fn random_unit() -> Vec3 {
    let z = rand::random::<f32>() * 2.0 - 1.0;
    let a = rand::random::<f32>() * TAU;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * a.cos(), z, r * a.sin())
}

fn rand_range(lo: f32, hi: f32) -> f32 {
    lo + rand::random::<f32>() * (hi - lo)
}

fn rand_sign() -> f32 {
    if rand::random::<bool>() {
        1.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tour_stays_smooth_and_cycles_phases() {
        let planet = Planet::new(12345);
        let mut cam = Camera::new(&planet, Vec3::new(0.3, 0.4, 0.86).normalize());
        let mut tour = Tour::new(&cam, &planet);

        let dt = 1.0 / 30.0;
        let mut min_dist = f32::INFINITY;
        let mut max_dist = 0.0f32;
        let mut prev_focus = cam.focus;
        let mut max_focus_jump = 0.0f32;
        let mut moved = false;

        // ~6 minutes of touring.
        for _ in 0..10_800 {
            tour.update(dt, &planet, &mut cam);

            // Focus stays a unit vector; nothing is NaN.
            assert!((cam.focus.length() - 1.0).abs() < 1e-3, "focus not unit: {}", cam.focus.length());
            let (vp, _, eye) = cam.view_proj(&planet);
            assert!(eye.is_finite(), "eye not finite");
            for col in vp.to_cols_array() {
                assert!(col.is_finite(), "view_proj has non-finite value");
            }

            let d = cam.distance();
            assert!(d.is_finite() && d > 0.0, "bad distance {d}");
            min_dist = min_dist.min(d);
            max_dist = max_dist.max(d);

            let jump = cam.focus.angle_between(prev_focus);
            max_focus_jump = max_focus_jump.max(jump);
            if jump > 1e-5 {
                moved = true;
            }
            prev_focus = cam.focus;
        }

        // It descends near the surface (cruise) and climbs to travel altitude.
        assert!(min_dist < 5_000.0, "tour never got near the surface (min {min_dist})");
        assert!(max_dist > 0.1 * PLANET_RADIUS, "tour never zoomed out to travel (max {max_dist})");
        // It moves, but never teleports — frame-to-frame motion stays small/smooth.
        assert!(moved, "tour camera never moved");
        assert!(max_focus_jump < 0.02, "tour made a jarring jump: {max_focus_jump} rad/frame");
    }
}
