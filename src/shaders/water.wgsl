// Ocean: a sea-level sphere with sin-wave displacement, fresnel rim, a sun
// specular highlight, and transparency. Drawn after terrain with alpha blending.

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,   // xyz, w = time
    sun_dir: vec4<f32>,
    params: vec4<f32>,
    atmosphere: vec4<f32>,
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
};

@vertex
fn vs(in: VsIn) -> VsOut {
    let t = g.camera_pos.w;
    let dir = normalize(in.pos);
    // A few overlapping sine waves for a gentle moving swell.
    let w1 = sin(dir.x * 38.0 + t * 1.3);
    let w2 = sin(dir.z * 31.0 + t * 1.7);
    let w3 = sin((dir.x + dir.y + dir.z) * 52.0 - t * 2.1);
    let wave = (w1 + w2 + w3) * 0.5;
    let p = in.pos + dir * wave;

    var out: VsOut;
    out.clip = g.view_proj * vec4<f32>(p, 1.0);
    out.world = p;
    out.normal = dir;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let v = normalize(g.camera_pos.xyz - in.world);
    let l = normalize(g.sun_dir.xyz);
    let h = normalize(l + v);

    let fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    let spec = pow(max(dot(n, h), 0.0), 120.0) * 1.3;

    let deep = vec3<f32>(0.015, 0.11, 0.26);
    let shallow = vec3<f32>(0.0, 0.34, 0.5);
    var col = mix(deep, shallow, clamp(fres * 0.7, 0.0, 1.0));
    col = col * (0.35 + 0.65 * max(dot(n, l), 0.0));
    col = col + g.atmosphere.rgb * fres * 0.35 + vec3<f32>(spec);

    // Distance fog so the far ocean blends into the horizon haze.
    let dist = length(in.world - g.camera_pos.xyz);
    let d = dist * g.params.x;
    let f = clamp(1.0 - exp(-d * d), 0.0, 1.0);
    col = mix(col, g.atmosphere.rgb * 0.85, f);

    let alpha = clamp(0.70 + fres * 0.28, 0.0, 1.0);
    return vec4<f32>(col, alpha);
}
