// Sky: a fullscreen pass that draws a seeded starfield and an atmospheric rim
// glow around the planet's silhouette. Rendered first, with no depth, so terrain
// and water draw over it.

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,   // xyz, w = time
    sun_dir: vec4<f32>,
    params: vec4<f32>,       // x = fog, y = planet radius, z = sea level, w = altitude
    atmosphere: vec4<f32>,   // rgb = tint, w = star seed
};
@group(0) @binding(0) var<uniform> g: Globals;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VsOut {
    // One oversized triangle covering the screen.
    var pts = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VsOut;
    let p = pts[vid];
    out.clip = vec4<f32>(p, 1.0, 1.0);
    out.ndc = p;
    return out;
}

fn hash31(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

fn star_field(dir: vec3<f32>, seed: f32) -> vec3<f32> {
    // Bucket the direction into cells; a sparse subset of cells hold a star.
    let s = dir * 260.0 + vec3<f32>(seed);
    let cell = floor(s);
    let r0 = hash31(cell);
    let r1 = hash31(cell + 11.3);
    let r2 = hash31(cell + 23.7);
    let present = step(0.992, r0);
    // Position the star within the cell and fall off with distance.
    let center = cell + vec3<f32>(r1, r2, hash31(cell + 5.1));
    let dist = length(s - center);
    let glow = present * smoothstep(0.9, 0.0, dist) * (0.4 + 0.6 * r1);
    // Slight color variation, tinted toward the atmosphere hue.
    let tint = mix(vec3<f32>(1.0), g.atmosphere.rgb, 0.25 * r2);
    return tint * glow;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct the world-space view ray for this pixel.
    let far = g.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let world = far.xyz / far.w;
    let dir = normalize(world - g.camera_pos.xyz);

    // Deep-space background with a faint atmosphere wash.
    var col = vec3<f32>(0.008, 0.010, 0.020) + g.atmosphere.rgb * 0.008;
    col = col + star_field(dir, g.atmosphere.w);

    // Atmospheric rim: glow where the view ray grazes the planet's limb.
    let c = g.camera_pos.xyz;
    let radius = g.params.y;
    let tc = -dot(c, dir);
    if (tc > 0.0) {
        let closest = length(c + dir * tc);
        if (closest > radius) {
            let glow = exp(-(closest - radius) / (radius * 0.07));
            // Brighter on the sunlit side.
            let sun_face = max(dot(dir, normalize(g.sun_dir.xyz)), 0.0) * 0.6 + 0.5;
            col = col + g.atmosphere.rgb * glow * 1.5 * sun_face;
        }
    }

    return vec4<f32>(col, 1.0);
}
