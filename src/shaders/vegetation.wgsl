// Vegetation: instanced low-poly plants. Each species' base mesh is uploaded once
// and drawn many times with a per-instance model matrix + colour tint, so a chunk
// stores ~80 bytes per plant instead of a baked copy of its whole geometry. Lit and
// fogged exactly like the terrain (but no water — plants are never sea).

const FOG_COLOR_SCALE: f32 = 0.85; // fog tint = atmosphere * this
const SKY_FILL: f32 = 0.12;        // soft ambient fill from straight up

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
    let up = normalize(in.world); // geocentric up at the plant (for the sky fill)
    let l = normalize(g.sun_dir.xyz);
    let amb = g.sun_dir.w;
    let diff = max(dot(n, l), 0.0);
    let sky_fill = max(dot(up, vec3<f32>(0.0, 1.0, 0.0)), 0.0) * SKY_FILL;
    let col = in.color * (amb + diff * (1.0 - amb) + sky_fill);
    return vec4<f32>(apply_fog(col, in.world), 1.0);
}
