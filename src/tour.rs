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
use crate::planet::{self, Biome, Planet, PLANET_RADIUS};
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

const DESCEND_DUR: f32 = 12.0; // a calmer settle, since the cruise altitude is low

// Cruise (low flyover): randomized per leg so it never feels mechanical.
// Target a low *altitude* (render units; ×10 = metres) so trees are clearly
// visible; the focus distance is derived from the tilt each leg, since the eye's
// height above terrain ≈ distance · cos(tilt). The camera's ground guard lifts it
// over the occasional tall rise.
const CRUISE_ALT_MIN: f32 = 10.0; // 100 m — low pass, trees clearly visible
const CRUISE_ALT_MAX: f32 = 22.0; // 220 m — varied, higher scenic passes

// Terrain-following: the cruise keeps the eye ~cruise_alt above a *smoothed* local
// ground so it never flies through hills, but only very loosely (it doesn't rigidly
// hug the surface). It climbs over rising ground fairly quickly and settles back
// slowly, which reads as a gentle, natural drift.
const GROUND_SAMPLE_REACH: f32 = 6.0; // sample terrain this far around the focus (render units)
const GROUND_RISE_TAU: f32 = 0.8; // smoothing time-constant when climbing (s)
const GROUND_FALL_TAU: f32 = 3.5; // slower coming back down — keeps it loose
const CRUISE_TILT_MIN: f32 = 0.85; // oblique, scenic horizon angles (rad)
const CRUISE_TILT_MAX: f32 = 1.18;
const CRUISE_DUR_MIN: f32 = 24.0; // seconds spent exploring a biome
const CRUISE_DUR_MAX: f32 = 38.0;
// Great-circle focus drift. At the low cruise altitude this is the ground speed
// you feel, so it's much gentler than the old high-altitude pan: rate × radius ×
// 10 m ≈ 19–51 m/s ground speed (a low, scenic flyover, not a hypersonic blur).
const CRUISE_PAN_MIN: f32 = 0.000_003; // rad/s
const CRUISE_PAN_MAX: f32 = 0.000_008;
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
// Random points probed per destination search. Bounded so even a leg that has to
// skip several absent biomes stays a one-frame cost at a phase transition (never
// the per-frame hot path). One probe ≈ one `Planet::sample`.
const DEST_SAMPLES: usize = 500;

/// Biomes the guided tour seeks out, one per leg, in cycle order — a deliberately
/// varied progression (lush → arid → cold → high → frozen). Ocean is omitted: the
/// tour is a low *land* flyover, and open water is merely the backdrop it crosses
/// between stops. A biome absent from (or, this leg, out of reach on) a given world
/// is skipped without stalling the cycle (see [`Tour::pick_biome_destination`]).
const BIOME_TOUR: [Biome; 10] = [
    Biome::TropicalForest,
    Biome::Grassland,
    Biome::TemperateForest,
    Biome::Desert,
    Biome::BorealForest,
    Biome::Tundra,
    Biome::Beach,
    Biome::Mountain,
    Biome::Snow,
    Biome::PolarIce,
];

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
    /// Cursor into [`BIOME_TOUR`]: the next biome the tour will seek out. Advances
    /// every biome it tries (found or skipped), so the cycle never stalls.
    biome_cursor: usize,
    /// Set when the biome cycle wraps past its end; consumed at the next cruise end
    /// to mark the tour finished (so the last biome is fully cruised first).
    lap_pending: bool,
    /// True once a full lap of [`BIOME_TOUR`] has been cruised — the cue for the
    /// `--video` recorder to play its space-pullback finale.
    finished: bool,

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

    // Smoothed terrain-following for the low cruise.
    cruise_alt: f32, // target eye height above ground (render units)
    ground_r: f32,   // smoothed local ground radius the eye stays above
}

impl Tour {
    /// Human-readable current phase, for the `--video` progress log.
    pub fn phase_label(&self) -> &'static str {
        match self.phase {
            Phase::Travel => "Travel",
            Phase::Descend => "Descend",
            Phase::Cruise => "Cruise",
        }
    }

    /// True once the tour has cruised one full lap of [`BIOME_TOUR`] — the cue for the
    /// `--video` recorder to take over with its space-pullback finale and finalize.
    /// (In the live app the tour just keeps drifting; nothing reads this.)
    pub fn toured_all_biomes(&self) -> bool {
        self.finished
    }

    /// Begin a tour from the camera's current view.
    pub fn new(cam: &Camera, planet: &Planet) -> Self {
        let mut tour = Self {
            phase: Phase::Travel,
            t: 0.0,
            dur: 1.0,
            biome_cursor: 0,
            lap_pending: false,
            finished: false,
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
            cruise_alt: CRUISE_ALT_MIN,
            ground_r: planet.surface_radius(cam.focus),
        };
        tour.begin_travel(cam.focus, cam.distance(), cam.heading(), cam.tilt(), planet);
        tour
    }

    pub fn update(&mut self, dt: f32, planet: &Planet, cam: &mut Camera) {
        self.t += dt;
        match self.phase {
            Phase::Travel | Phase::Descend => {
                let s = planet::smoothstep(0.0, 1.0, (self.t / self.dur).min(1.0));
                // Exponential (multiplicative) zoom feels natural across scales.
                let dist = self.s_dist * (self.e_dist / self.s_dist).powf(s);
                // Lateral/heading progress tracks ALTITUDE, not time: the camera
                // barely moves sideways while low and crosses quickly once it has
                // zoomed out. So leaving a treetop flyby it climbs out and speeds up
                // smoothly instead of suddenly, and arriving it decelerates as it
                // drops in. (On the descend leg the focus is stationary, so this
                // only shapes the heading settle.)
                let lat = if (self.e_dist - self.s_dist).abs() < 1e-3 {
                    s
                } else {
                    ((dist - self.s_dist) / (self.e_dist - self.s_dist)).clamp(0.0, 1.0)
                };
                let focus = slerp_dir(self.s_focus, self.e_focus, lat);
                let head = self.s_head + shortest(self.e_head - self.s_head) * lat;
                let tilt = self.s_tilt + (self.e_tilt - self.s_tilt) * s;
                cam.set_view(focus, dist, head, tilt);

                if self.t >= self.dur {
                    if self.phase == Phase::Travel {
                        self.begin_descend(planet);
                    } else {
                        self.begin_cruise(planet);
                    }
                }
            }
            Phase::Cruise => {
                // Drift the focus slowly along a great circle and pan the view.
                self.c_focus = (Quat::from_axis_angle(self.drift_axis, self.pan_rate * dt) * self.c_focus).normalize();
                self.c_head += self.head_rate * dt;
                let tilt = (self.base_tilt + (self.t * self.tilt_freq).sin() * self.tilt_amp).clamp(0.0, CRUISE_TILT_CLAMP);

                // Loosely follow the terrain: keep the eye ~cruise_alt above a
                // smoothed local-max ground. Climb fairly quickly over rising land
                // (so it never flies through hills) and settle back slowly.
                let focus_r = planet.surface_radius(self.c_focus);
                let target_r = local_ground_radius(planet, self.c_focus, GROUND_SAMPLE_REACH);
                let tau = if target_r > self.ground_r { GROUND_RISE_TAU } else { GROUND_FALL_TAU };
                self.ground_r += (target_r - self.ground_r) * (1.0 - (-dt / tau).exp());
                let ref_r = self.ground_r.max(focus_r);
                self.c_dist = (ref_r + self.cruise_alt - focus_r) / tilt.cos();
                cam.set_view(self.c_focus, self.c_dist, self.c_head, tilt);

                if self.t >= self.dur {
                    if self.lap_pending {
                        // Cruised every biome in the cycle once — hold in this gentle
                        // drift; the video recorder takes over for the finale (the live
                        // app ignores `finished`, so it keeps drifting indefinitely).
                        self.finished = true;
                    } else {
                        self.begin_travel(self.c_focus, self.c_dist, self.c_head, self.base_tilt, planet);
                    }
                }
            }
        }
    }

    fn begin_travel(&mut self, from_focus: Vec3, from_dist: f32, from_head: f32, from_tilt: f32, planet: &Planet) {
        let dest = self.pick_biome_destination(from_focus, planet);
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
        // Settle into a low flyover: pick the tilt, then derive the focus distance
        // that puts the eye at the target altitude (altitude ≈ dist · cos(tilt)).
        let cruise_tilt = rand_range(CRUISE_TILT_MIN, CRUISE_TILT_MAX);
        let cruise_alt = rand_range(CRUISE_ALT_MIN, CRUISE_ALT_MAX);
        self.cruise_alt = cruise_alt;
        let cruise_dist = cruise_alt / cruise_tilt.cos();
        let arrive_head = self.e_head + rand_range(-ARRIVE_HEADING_JITTER, ARRIVE_HEADING_JITTER);
        self.phase = Phase::Descend;
        self.t = 0.0;
        self.dur = DESCEND_DUR;
        // Start from where Travel ended; settle onto the biome.
        self.set_segment(self.e_focus, self.e_dist, self.e_head, self.e_tilt, self.e_focus, cruise_dist, arrive_head, cruise_tilt);
    }

    fn begin_cruise(&mut self, planet: &Planet) {
        self.phase = Phase::Cruise;
        self.t = 0.0;
        self.dur = rand_range(CRUISE_DUR_MIN, CRUISE_DUR_MAX);
        self.c_focus = self.e_focus;
        self.c_dist = self.e_dist;
        self.c_head = self.e_head;
        // Start the terrain-follow reference at the focus ground; it eases up to
        // clear nearby rises over the first second of cruise (no jump on arrival).
        self.ground_r = planet.surface_radius(self.c_focus);

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

    /// Advance the biome cycle and return a scenic land point in the next biome
    /// that occurs on this planet. Every biome it tries advances the cursor, so
    /// biomes absent from — or, this leg, out of reach on — a world are skipped
    /// rather than stalling the tour; because the camera roams a little each leg,
    /// every biome the planet does have is eventually reached. Falls back to any
    /// land, then anywhere, on a near-waterworld with nothing in range.
    fn pick_biome_destination(&mut self, from: Vec3, planet: &Planet) -> Vec3 {
        for _ in 0..BIOME_TOUR.len() {
            let biome = BIOME_TOUR[self.biome_cursor];
            self.biome_cursor = (self.biome_cursor + 1) % BIOME_TOUR.len();
            if self.biome_cursor == 0 {
                self.lap_pending = true; // wrapped — every biome attempted once this lap
            }
            if let Some(dest) = pick_destination(from, planet, Some(biome)) {
                tracing::info!(biome = biome.name(), "tour: cruising to next biome");
                return dest;
            }
        }
        pick_destination(from, planet, None).unwrap_or_else(random_unit)
    }
}

/// Max terrain radius over the focus and four neighbours `reach` units away — the
/// local ground the cruise camera should stay above so it doesn't clip a nearby rise.
fn local_ground_radius(planet: &Planet, focus: Vec3, reach: f32) -> f32 {
    let (north, east) = frame(focus);
    let ang = reach / PLANET_RADIUS; // small-angle tangent offset
    let mut r = planet.surface_radius(focus);
    for t in [north, -north, east, -east] {
        let dir = (focus + t * ang).normalize();
        r = r.max(planet.surface_radius(dir));
    }
    r
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

/// Probe up to [`DEST_SAMPLES`] random surface points for a scenic destination a
/// comfortable arc from `current` and above the coast. With `Some(biome)`, only
/// that biome qualifies; with `None`, any solid land does. `None` (the return)
/// means nothing matched within the budget.
fn pick_destination(current: Vec3, planet: &Planet, target: Option<Biome>) -> Option<Vec3> {
    for _ in 0..DEST_SAMPLES {
        let d = random_unit();
        let arc = current.angle_between(d);
        if !(DEST_ARC_MIN..DEST_ARC_MAX).contains(&arc) {
            continue;
        }
        let s = planet.sample(d);
        if s.height <= DEST_MIN_HEIGHT {
            continue; // skip ocean and the shoreline — keep the cruise on solid land
        }
        if target.is_some_and(|biome| s.biome != biome) {
            continue; // wrong biome for this leg
        }
        return Some(d);
    }
    None
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
    fn tour_finishes_after_touring_all_biomes() {
        let planet = Planet::new(12345);
        let mut cam = Camera::new(&planet, Vec3::new(0.3, 0.4, 0.86).normalize());
        let mut tour = Tour::new(&cam, &planet);
        let dt = 1.0 / 30.0;
        // Cap well above a full lap (~10 biomes × ~70 s) so a bug can't hang the test.
        let max_steps = 30 * 60 * 30; // 30 minutes of sim
        let mut steps = 0;
        while !tour.toured_all_biomes() && steps < max_steps {
            tour.update(dt, &planet, &mut cam);
            steps += 1;
        }
        assert!(tour.toured_all_biomes(), "tour never reported completion within 30 min of sim");
        // Not instant: it actually cruised biomes first (guards a premature-finish bug).
        assert!(steps as f32 * dt > 30.0, "tour finished implausibly fast ({steps} steps)");
    }

    #[test]
    fn tour_stays_smooth_and_cycles_phases() {
        let planet = Planet::new(12345);
        let mut cam = Camera::new(&planet, Vec3::new(0.3, 0.4, 0.86).normalize());
        let mut tour = Tour::new(&cam, &planet);

        let dt = 1.0 / 30.0;
        let mut min_dist = f32::INFINITY;
        let mut max_dist = 0.0f32;
        let mut min_clear = f32::INFINITY; // eye height above the terrain beneath it
        let mut prev_focus = cam.focus;
        let mut max_focus_jump = 0.0f32;
        let mut moved = false;
        // Distinct biomes visited at cruise altitude — the tour should seek out
        // variety, not loiter in one. (`Biome` isn't `Hash`; key by discriminant.)
        let mut cruise_biomes = std::collections::HashSet::new();

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

            // The eye must never sink into the terrain beneath it.
            let clear = eye.length() - planet.surface_radius(eye.normalize());
            assert!(clear > 0.0, "eye went underground (clearance {clear})");
            min_clear = min_clear.min(clear);

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

            // Only the low cruise reflects the targeted biome (travel sweeps over
            // many); travel/orbit altitude is ~0.16·radius, cruise is tens of units.
            if cam.distance() < 1_000.0 {
                cruise_biomes.insert(planet.sample(cam.focus).biome as usize);
            }
        }

        // It descends near the surface (cruise) and climbs to travel altitude.
        assert!(min_dist < 5_000.0, "tour never got near the surface (min {min_dist})");
        assert!(max_dist > 0.1 * PLANET_RADIUS, "tour never zoomed out to travel (max {max_dist})");
        // The low cruise genuinely flies low (within ~150 m of the ground) without
        // ever clipping through it.
        assert!(min_clear < 16.0, "tour never flew low over the terrain (min clearance {min_clear})");
        // It moves, but never teleports — frame-to-frame motion stays small/smooth.
        assert!(moved, "tour camera never moved");
        assert!(max_focus_jump < 0.02, "tour made a jarring jump: {max_focus_jump} rad/frame");
        // The biome cycle takes it through several distinct biomes over 6 minutes,
        // not round and round one stretch of land.
        assert!(cruise_biomes.len() >= 3, "tour lacked biome variety: {} distinct cruise biomes", cruise_biomes.len());
    }
}
