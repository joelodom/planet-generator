// Terrain: vertex-colored, diffuse-lit, distance-fogged.

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,   // xyz, w = time
    sun_dir: vec4<f32>,      // xyz = dir to sun, w = ambient
    params: vec4<f32>,       // x = fog density, y = planet radius, z = sea level, w = altitude
    atmosphere: vec4<f32>,   // rgb = tint, w = unused
};
@group(0) @binding(0) var<uniform> g: Globals;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = g.view_proj * vec4<f32>(in.pos, 1.0);
    out.world = in.pos;
    out.normal = in.normal;
    out.color = in.color;
    return out;
}

fn apply_fog(color: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let dist = length(world - g.camera_pos.xyz);
    let d = dist * g.params.x;
    let f = clamp(1.0 - exp(-d * d), 0.0, 1.0);
    let fog_color = g.atmosphere.rgb * 0.85;
    return mix(color, fog_color, f);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(g.sun_dir.xyz);
    let amb = g.sun_dir.w;
    let diff = max(dot(n, l), 0.0);
    // Soft sky fill from straight up keeps shadowed slopes from going black.
    let sky_fill = max(dot(n, vec3<f32>(0.0, 1.0, 0.0)), 0.0) * 0.12;
    let lit = in.color * (amb + diff * (1.0 - amb) + sky_fill);
    return vec4<f32>(apply_fog(lit, in.world), 1.0);
}
