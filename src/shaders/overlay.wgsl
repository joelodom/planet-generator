// Screen-space overlay: instanced solid-color quads in normalized device
// coordinates. Used to draw the help panel — a dim backdrop, a panel, and one
// tiny quad per lit bitmap-font pixel. No camera, no textures: fully portable.

struct VsIn {
    @location(0) corner: vec2<f32>,  // unit quad corner, 0..1
    @location(1) rect: vec4<f32>,    // xy = top-left (NDC), zw = size (NDC, h negative-down)
    @location(2) color: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    let p = in.rect.xy + in.corner * in.rect.zw;
    out.clip = vec4<f32>(p, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
