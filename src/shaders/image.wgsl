// Textured screen-space quad for the help overlay's planet image. Samples a
// texture over a unit-quad rect and applies a soft circular mask so the planet
// reads as a clean disc rather than a square with black corners.

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

const DISC_EDGE_INNER: f32 = 0.47; // circular mask: opaque within this radius ...
const DISC_EDGE_OUTER: f32 = 0.5;  // ... fading to transparent by the quad edge

struct VsIn {
    @location(0) corner: vec2<f32>, // unit quad 0..1
    @location(1) rect: vec4<f32>,   // xy = top-left (NDC), zw = size (NDC, h negative)
    @location(2) tint: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = vec4<f32>(in.rect.xy + in.corner * in.rect.zw, 0.0, 1.0);
    out.uv = in.corner;
    out.tint = in.tint;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Soft circular mask (anti-aliased edge).
    let d = distance(in.uv, vec2<f32>(0.5, 0.5));
    let alpha = 1.0 - smoothstep(DISC_EDGE_INNER, DISC_EDGE_OUTER, d);
    if (alpha <= 0.0) {
        discard;
    }
    let c = textureSample(tex, samp, in.uv);
    return vec4<f32>(c.rgb * in.tint.rgb, alpha * in.tint.a);
}
