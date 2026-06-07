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
    let radial = normalize(in.normal);
    let l = normalize(g.sun_dir.xyz);
    let v = normalize(g.camera_pos.xyz - in.world);
    let amb = g.sun_dir.w;

    // The ocean is part of this mesh, sitting at exactly sea level (= planet
    // radius, params.y). Detect it to add a moving sun glint and rim sheen.
    let is_water = length(in.world) < g.params.y + 1.5;

    var n = radial;
    if (is_water) {
        // Gentle animated ripples perturb the (radial) normal for a live glint.
        let t = g.camera_pos.w;
        let p = in.world;
        let ripple = vec3<f32>(
            sin(p.x * 0.7 + t * 1.4) + sin(p.z * 1.1 - t * 1.0),
            0.0,
            cos(p.z * 0.9 + t * 1.2) + sin(p.x * 0.6 - t * 0.8),
        ) * 0.03;
        n = normalize(radial + ripple);
    }

    let diff = max(dot(n, l), 0.0);
    let sky_fill = max(dot(radial, vec3<f32>(0.0, 1.0, 0.0)), 0.0) * 0.12;
    var col = in.color * (amb + diff * (1.0 - amb) + sky_fill);

    if (is_water) {
        let h = normalize(l + v);
        let spec = pow(max(dot(n, h), 0.0), 90.0) * 1.6;
        let fres = pow(1.0 - max(dot(radial, v), 0.0), 4.0);
        col = col + vec3<f32>(spec) + g.atmosphere.rgb * fres * 0.25;
    }

    return vec4<f32>(apply_fog(col, in.world), 1.0);
}
