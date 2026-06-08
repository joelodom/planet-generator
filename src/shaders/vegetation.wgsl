// Instanced vegetation: per-instance model matrix + color tint, lit and fogged
// the same way as terrain so plants sit naturally in the scene.

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    params: vec4<f32>,
    atmosphere: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

const FOG_COLOR_SCALE: f32 = 0.85; // fog tint = atmosphere * this (matches terrain)

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    // Instance: a mat4 split across four vec4 locations, plus a color tint.
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
    let world = model * vec4<f32>(in.pos, 1.0);
    // Rotation only (uniform scale) — fine to use the upper 3x3 for normals.
    let n = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);

    var out: VsOut;
    out.clip = g.view_proj * world;
    out.world = world.xyz;
    out.normal = n;
    out.color = in.color * in.tint.rgb;
    return out;
}

fn apply_fog(color: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let dist = length(world - g.camera_pos.xyz);
    let d = dist * g.params.x;
    let f = clamp(1.0 - exp(-d * d), 0.0, 1.0);
    return mix(color, g.atmosphere.rgb * FOG_COLOR_SCALE, f);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(g.sun_dir.xyz);
    let amb = g.sun_dir.w;
    let diff = max(dot(n, l), 0.0);
    let lit = in.color * (amb + diff * (1.0 - amb));
    return vec4<f32>(apply_fog(lit, in.world), 1.0);
}
