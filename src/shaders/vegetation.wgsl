// Vegetation: instanced low-poly plants. Each species' base mesh is uploaded once
// and drawn many times with a per-instance model matrix + colour tint, so a chunk
// stores ~80 bytes per plant instead of a baked copy of its whole geometry. Lit and
// fogged exactly like the terrain (but no water — plants are never sea).

const FOG_COLOR_SCALE: f32 = 0.85; // fog tint = atmosphere * this
const SKY_FILL: f32 = 0.12;        // soft ambient fill for upward-facing surfaces
const WRAP_FACTOR: f32 = 0.6;      // half-Lambert: 1 = hard Lambert, lower = softer wrap
const TRANSLUCENCY: f32 = 0.7;     // strength of sun transmitted through back-lit foliage
const TRANS_FLOOR: f32 = 0.3;      // back-light seen from any angle; the rest needs looking sunward
const SHEEN_POWER: f32 = 16.0;     // waxy leaf highlight tightness (low = broad/soft)
const SHEEN_STRENGTH: f32 = 0.15;  // waxy leaf highlight strength

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,   // xyz, w = time
    sun_dir: vec4<f32>,      // xyz = dir to sun, w = ambient
    params: vec4<f32>,       // x = fog density, y = planet radius, z = sea level, w = altitude
    atmosphere: vec4<f32>,   // rgb = tint
};
@group(0) @binding(0) var<uniform> g: Globals;

struct VsIn {
    // Base-mesh vertex (local space).
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    // Per-instance: the four columns of the local→world model matrix, plus a tint.
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
    @location(7) tint: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = (model * vec4<f32>(in.pos, 1.0)).xyz;
    // Uniform scale + rotation, so the upper 3x3 (re-normalised in the fragment)
    // carries the normal correctly.
    let nmat = mat3x3<f32>(in.m0.xyz, in.m1.xyz, in.m2.xyz);
    var out: VsOut;
    out.clip = g.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = nmat * in.normal;
    out.color = in.color * in.tint.rgb;
    return out;
}

fn apply_fog(color: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let dist = length(world - g.camera_pos.xyz);
    let d = dist * g.params.x;
    let f = clamp(1.0 - exp(-d * d), 0.0, 1.0);
    let fog_color = g.atmosphere.rgb * FOG_COLOR_SCALE;
    return mix(color, fog_color, f);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let up = normalize(in.world);                 // geocentric up at the plant
    let l = normalize(g.sun_dir.xyz);
    let v = normalize(g.camera_pos.xyz - in.world);
    let amb = g.sun_dir.w;

    // Half-Lambert wrap: foliage bounces a lot of light, so soften the terminator
    // instead of cutting hard to black on the shaded side.
    let ndl = dot(n, l);
    let wrap = clamp(ndl * WRAP_FACTOR + (1.0 - WRAP_FACTOR), 0.0, 1.0);

    // Hemispheric sky fill: upward-facing surfaces catch sky light.
    let sky_fill = max(dot(n, up), 0.0) * SKY_FILL;

    // Subsurface back-light: sun transmitted through thin foliage whose far side faces
    // the sun. Tinted by the surface's own colour, so it self-gates (bright leaves glow,
    // dark bark barely transmits) and is strongest when looking toward the sun.
    let back = max(dot(-n, l), 0.0);
    let sunward = max(dot(v, -l), 0.0);
    let trans = TRANSLUCENCY * back * (TRANS_FLOOR + (1.0 - TRANS_FLOOR) * sunward);

    var col = in.color * (amb + wrap * (1.0 - amb) + sky_fill + trans);

    // Weak, broad waxy sheen on the sunlit side.
    let h = normalize(l + v);
    let sheen = pow(max(dot(n, h), 0.0), SHEEN_POWER) * SHEEN_STRENGTH * max(ndl, 0.0);
    col = col + vec3<f32>(sheen);

    return vec4<f32>(apply_fog(col, in.world), 1.0);
}
