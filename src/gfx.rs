//! The renderer: all wgpu setup and the per-frame draw of sky, terrain (the
//! ocean is part of the terrain mesh), and vegetation into one depth-tested pass.
//!
//! The renderer owns the GPU-resident chunk cache (keyed by [`ChunkKey`]). The
//! main loop hands it freshly meshed [`CpuChunk`]s to upload and a list of which
//! chunks to draw; it knows nothing about LOD policy or planet maths. That keeps
//! the rendering layer a thin, replaceable slab beneath the simulation.

use crate::lod::ChunkKey;
use crate::mesh::{self, CpuChunk, InstanceRaw, Vertex};
use crate::overlay::{self, OverlayInstance};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Uniform block shared by every shader. Mirrors the `Globals` struct in WGSL.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub params: [f32; 4],
    pub atmosphere: [f32; 4],
}

/// An indexed mesh living on the GPU.
struct GpuMesh {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    count: u32,
}

impl GpuMesh {
    fn upload(device: &wgpu::Device, verts: &[Vertex], indices: &[u32]) -> Self {
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh-verts"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh-indices"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self { vbuf, ibuf, count: indices.len() as u32 }
    }
}

/// An instance buffer (vegetation placements for one chunk).
struct InstanceBuf {
    buf: wgpu::Buffer,
    count: u32,
}

impl InstanceBuf {
    fn upload(device: &wgpu::Device, instances: &[InstanceRaw]) -> Option<Self> {
        if instances.is_empty() {
            return None;
        }
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Some(Self { buf, count: instances.len() as u32 })
    }
}

/// A terrain chunk plus its vegetation, all GPU-resident.
struct GpuChunk {
    terrain: GpuMesh,
    trees: Option<InstanceBuf>,
    shrubs: Option<InstanceBuf>,
}

const VERT_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];
const INST_ATTRS: [wgpu::VertexAttribute; 5] =
    wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4];

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERT_ATTRS,
    }
}

fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INST_ATTRS,
    }
}

// Screen-space overlay (help panel) — a unit quad instanced per colored rect.
const OVERLAY_CORNERS: [[f32; 2]; 6] =
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const OVERLAY_CORNER_ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];
const OVERLAY_INST_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4];

fn overlay_corner_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<[f32; 2]>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &OVERLAY_CORNER_ATTRS,
    }
}

fn overlay_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<OverlayInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &OVERLAY_INST_ATTRS,
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: (u32, u32),

    depth_view: wgpu::TextureView,

    globals_buf: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,

    sky_pipeline: wgpu::RenderPipeline,
    terrain_pipeline: wgpu::RenderPipeline,
    terrain_wire: Option<wgpu::RenderPipeline>,
    veg_pipeline: wgpu::RenderPipeline,

    tree_mesh: GpuMesh,
    shrub_mesh: GpuMesh,

    overlay_pipeline: wgpu::RenderPipeline,
    overlay_quad: wgpu::Buffer,
    overlay_instances: Option<(wgpu::Buffer, u32)>,
    overlay_lines: Vec<String>,
    overlay_highlight: usize,
    pub overlay_visible: bool,

    // Planet image shown in the help overlay.
    image_pipeline: wgpu::RenderPipeline,
    planet_bind: wgpu::BindGroup,
    image_instance: Option<wgpu::Buffer>,

    chunks: HashMap<ChunkKey, GpuChunk>,
    pub wireframe: bool,
    pub supports_wireframe: bool,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let size = (size.width.max(1), size.height.max(1));

        let mut idesc = wgpu::InstanceDescriptor::new_without_display_handle();
        // Cross-platform: Metal (macOS), DX12/Vulkan (Windows, incl. the planned
        // RTX 5090 box), Vulkan/GL (Linux).
        idesc.backends = wgpu::Backends::METAL
            | wgpu::Backends::DX12
            | wgpu::Backends::VULKAN
            | wgpu::Backends::GL;
        let instance = wgpu::Instance::new(idesc);
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await?;

        let info = adapter.get_info();
        tracing::info!(
            name = %info.name,
            backend = ?info.backend,
            device_type = ?info.device_type,
            driver = %info.driver,
            "gpu adapter selected"
        );

        let supports_wireframe = adapter.features().contains(wgpu::Features::POLYGON_MODE_LINE);
        let required_features = if supports_wireframe {
            wgpu::Features::POLYGON_MODE_LINE
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("planet-device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        tracing::debug!(
            format = ?format,
            present_mode = ?config.present_mode,
            size = ?(size.0, size.1),
            supports_wireframe,
            "surface configured"
        );

        let depth_view = create_depth(&device, size.0, size.1);

        // Globals uniform + bind group (group 0, binding 0) used by all shaders.
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals-bind"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let sky_sh = shader(&device, "sky", include_str!("shaders/sky.wgsl"));
        let terrain_sh = shader(&device, "terrain", include_str!("shaders/terrain.wgsl"));
        let veg_sh = shader(&device, "vegetation", include_str!("shaders/vegetation.wgsl"));

        let sky_pipeline = make_pipeline(&device, &pipeline_layout, &sky_sh, &[], format, PassKind::Sky, wgpu::PolygonMode::Fill);
        let terrain_pipeline = make_pipeline(&device, &pipeline_layout, &terrain_sh, &[vertex_layout()], format, PassKind::Opaque, wgpu::PolygonMode::Fill);
        let terrain_wire = if supports_wireframe {
            Some(make_pipeline(&device, &pipeline_layout, &terrain_sh, &[vertex_layout()], format, PassKind::Opaque, wgpu::PolygonMode::Line))
        } else {
            None
        };
        let veg_pipeline = make_pipeline(&device, &pipeline_layout, &veg_sh, &[vertex_layout(), instance_layout()], format, PassKind::Opaque, wgpu::PolygonMode::Fill);

        // Overlay pipeline: no bind groups (pure screen-space), alpha blended.
        let overlay_sh = shader(&device, "overlay", include_str!("shaders/overlay.wgsl"));
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let overlay_pipeline = make_pipeline(
            &device,
            &overlay_layout,
            &overlay_sh,
            &[overlay_corner_layout(), overlay_instance_layout()],
            format,
            PassKind::Overlay,
            wgpu::PolygonMode::Fill,
        );
        let overlay_quad = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay-quad"),
            contents: bytemuck::cast_slice(&OVERLAY_CORNERS),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Planet image (embedded PNG) → texture, sampler, bind group, pipeline.
        let (planet_view, planet_sampler) = load_planet_texture(&device, &queue);
        let planet_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let planet_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-bind"),
            layout: &planet_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&planet_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&planet_sampler) },
            ],
        });
        let image_sh = shader(&device, "image", include_str!("shaders/image.wgsl"));
        let image_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("image-layout"),
            bind_group_layouts: &[Some(&planet_bgl)],
            immediate_size: 0,
        });
        let image_pipeline = make_pipeline(
            &device,
            &image_layout,
            &image_sh,
            &[overlay_corner_layout(), overlay_instance_layout()],
            format,
            PassKind::Overlay,
            wgpu::PolygonMode::Fill,
        );

        let tm = mesh::tree_mesh();
        let sm = mesh::shrub_mesh();
        let tree_mesh = GpuMesh::upload(&device, &tm.vertices, &tm.indices);
        let shrub_mesh = GpuMesh::upload(&device, &sm.vertices, &sm.indices);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            depth_view,
            globals_buf,
            globals_bind,
            sky_pipeline,
            terrain_pipeline,
            terrain_wire,
            veg_pipeline,
            tree_mesh,
            shrub_mesh,
            overlay_pipeline,
            overlay_quad,
            overlay_instances: None,
            overlay_lines: Vec::new(),
            overlay_highlight: usize::MAX,
            overlay_visible: false,
            image_pipeline,
            planet_bind,
            image_instance: None,
            chunks: HashMap::new(),
            wireframe: false,
            supports_wireframe,
        })
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.size = (w, h);
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth(&self.device, w, h);
        if self.overlay_visible {
            self.rebuild_overlay();
        }
    }

    /// Set the help overlay's text (built once at startup).
    /// Set the overlay's text and which row (if any) is the highlighted setting.
    pub fn set_overlay(&mut self, lines: Vec<String>, highlight: usize) {
        self.overlay_lines = lines;
        self.overlay_highlight = highlight;
        if self.overlay_visible {
            self.rebuild_overlay();
        }
    }

    /// Show/hide the settings overlay, rebuilding its geometry when shown.
    pub fn set_overlay_visible(&mut self, visible: bool) {
        self.overlay_visible = visible;
        if visible {
            self.rebuild_overlay();
        }
    }

    fn rebuild_overlay(&mut self) {
        let geo = overlay::layout(&self.overlay_lines, self.overlay_highlight, self.size.0, self.size.1);
        self.overlay_instances = if geo.quads.is_empty() {
            None
        } else {
            let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("overlay-instances"),
                contents: bytemuck::cast_slice(&geo.quads),
                usage: wgpu::BufferUsages::VERTEX,
            });
            Some((buf, geo.quads.len() as u32))
        };
        self.image_instance = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay-image"),
            contents: bytemuck::bytes_of(&geo.image),
            usage: wgpu::BufferUsages::VERTEX,
        }));
    }

    pub fn has_chunk(&self, key: ChunkKey) -> bool {
        self.chunks.contains_key(&key)
    }

    pub fn upload_chunk(&mut self, key: ChunkKey, cpu: CpuChunk) {
        let terrain = GpuMesh::upload(&self.device, &cpu.vertices, &cpu.indices);
        let trees = InstanceBuf::upload(&self.device, &cpu.trees);
        let shrubs = InstanceBuf::upload(&self.device, &cpu.shrubs);
        self.chunks.insert(key, GpuChunk { terrain, trees, shrubs });
    }

    /// Drop chunks no longer needed, keeping memory bounded. Roots and anything
    /// in `keep` are retained.
    pub fn evict(&mut self, keep: &std::collections::HashSet<ChunkKey>, limit: usize) {
        if self.chunks.len() <= limit {
            return;
        }
        let roots = ChunkKey::roots();
        self.chunks
            .retain(|k, _| keep.contains(k) || roots.contains(k));
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Drop every resident chunk. Used when a detail setting baked into geometry
    /// changes, so the world re-streams at the new resolution.
    pub fn clear_chunks(&mut self) {
        self.chunks.clear();
    }

    pub fn update_globals(&self, g: &Globals) {
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(g));
    }

    pub fn render(&mut self, draw: &[ChunkKey]) {
        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match self.surface.get_current_texture() {
            Cst::Success(f) | Cst::Suboptimal(f) => f,
            Cst::Outdated | Cst::Lost => {
                tracing::debug!("surface lost/outdated; reconfiguring");
                self.surface.configure(&self.device, &self.config);
                return;
            }
            other => {
                // Timeout / Occluded / Validation: skip this frame.
                tracing::trace!(status = ?std::mem::discriminant(&other), "surface frame skipped");
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.01, g: 0.01, b: 0.02, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_bind_group(0, &self.globals_bind, &[]);

            // Sky first (depth-always, no write) to fill the background.
            pass.set_pipeline(&self.sky_pipeline);
            pass.draw(0..3, 0..1);

            // Terrain.
            let terrain_pipe = if self.wireframe {
                self.terrain_wire.as_ref().unwrap_or(&self.terrain_pipeline)
            } else {
                &self.terrain_pipeline
            };
            pass.set_pipeline(terrain_pipe);
            for key in draw {
                if let Some(chunk) = self.chunks.get(key) {
                    pass.set_vertex_buffer(0, chunk.terrain.vbuf.slice(..));
                    pass.set_index_buffer(chunk.terrain.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..chunk.terrain.count, 0, 0..1);
                }
            }

            // Vegetation (skip in wireframe mode to keep the debug view legible).
            if !self.wireframe {
                pass.set_pipeline(&self.veg_pipeline);
                for key in draw {
                    if let Some(chunk) = self.chunks.get(key) {
                        if let Some(trees) = &chunk.trees {
                            draw_instanced(&mut pass, &self.tree_mesh, trees);
                        }
                        if let Some(shrubs) = &chunk.shrubs {
                            draw_instanced(&mut pass, &self.shrub_mesh, shrubs);
                        }
                    }
                }
            }

            // (Ocean is part of the terrain mesh now — no separate water pass.)

            // Help overlay on top of everything (screen-space).
            if self.overlay_visible {
                if let Some((buf, count)) = &self.overlay_instances {
                    pass.set_pipeline(&self.overlay_pipeline);
                    pass.set_vertex_buffer(0, self.overlay_quad.slice(..));
                    pass.set_vertex_buffer(1, buf.slice(..));
                    pass.draw(0..6, 0..*count);
                }
                if let Some(img) = &self.image_instance {
                    pass.set_pipeline(&self.image_pipeline);
                    pass.set_bind_group(0, &self.planet_bind, &[]);
                    pass.set_vertex_buffer(0, self.overlay_quad.slice(..));
                    pass.set_vertex_buffer(1, img.slice(..));
                    pass.draw(0..6, 0..1);
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

fn draw_instanced<'a>(pass: &mut wgpu::RenderPass<'a>, base: &'a GpuMesh, inst: &'a InstanceBuf) {
    pass.set_vertex_buffer(0, base.vbuf.slice(..));
    pass.set_vertex_buffer(1, inst.buf.slice(..));
    pass.set_index_buffer(base.ibuf.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..base.count, 0, 0..inst.count);
}

/// Planet image shown in the help overlay, baked into the binary.
const PLANET_PNG: &[u8] = include_bytes!("../assets/planet.png");

/// Decode the embedded planet PNG and upload it as an sRGB texture.
fn load_planet_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::TextureView, wgpu::Sampler) {
    let (rgba, w, h) = decode_png(PLANET_PNG);
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planet-image"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        size,
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("planet-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (view, sampler)
}

/// Decode a PNG to tightly-packed RGBA8. Handles RGB and RGBA sources.
fn decode_png(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().expect("planet.png header");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("planet.png buffer size")];
    let info = reader.next_frame(&mut buf).expect("planet.png frame");
    let (w, h) = (info.width, info.height);
    let used = &buf[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => used.to_vec(),
        png::ColorType::Rgb => {
            let mut v = Vec::with_capacity((w * h * 4) as usize);
            for px in used.chunks_exact(3) {
                v.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            v
        }
        other => panic!("unsupported planet.png color type {other:?}"),
    };
    (rgba, w, h)
}

fn create_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn shader(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    })
}

/// How a pipeline interacts with the shared depth buffer and blending.
#[derive(Clone, Copy)]
enum PassKind {
    Sky,     // depth always, no write, opaque
    Opaque,  // depth less, write, opaque
    Overlay, // depth always, no write, alpha blend (screen-space UI on top)
}

fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    buffers: &[wgpu::VertexBufferLayout<'static>],
    format: wgpu::TextureFormat,
    kind: PassKind,
    polygon_mode: wgpu::PolygonMode,
) -> wgpu::RenderPipeline {
    let (depth_write, depth_compare) = match kind {
        PassKind::Sky | PassKind::Overlay => (false, wgpu::CompareFunction::Always),
        PassKind::Opaque => (true, wgpu::CompareFunction::Less),
    };
    let blend = match kind {
        PassKind::Overlay => Some(wgpu::BlendState::ALPHA_BLENDING),
        _ => Some(wgpu::BlendState::REPLACE),
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // terrain faces/skirts have mixed winding; don't cull
            unclipped_depth: false,
            polygon_mode,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod smoke {
    //! Headless GPU validation: build the *real* shaders and pipelines, render
    //! one chunk + sky + water + vegetation to an offscreen target inside a
    //! validation error scope, and assert nothing was rejected. Skips cleanly if
    //! no GPU adapter is present.
    use super::*;
    use crate::lod::ChunkKey;
    use crate::planet::Planet;
    use glam::Mat4;

    #[test]
    fn offscreen_pipeline_validates() {
        let mut idesc = wgpu::InstanceDescriptor::new_without_display_handle();
        // Cross-platform: Metal (macOS), DX12/Vulkan (Windows, incl. the planned
        // RTX 5090 box), Vulkan/GL (Linux).
        idesc.backends = wgpu::Backends::METAL
            | wgpu::Backends::DX12
            | wgpu::Backends::VULKAN
            | wgpu::Backends::GL;
        let instance = wgpu::Instance::new(idesc);
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })) {
            Ok(a) => a,
            Err(_) => {
                eprintln!("smoke: no GPU adapter available; skipping");
                return;
            }
        };
        let supports_wire = adapter.features().contains(wgpu::Features::POLYGON_MODE_LINE);
        let feats = if supports_wire { wgpu::Features::POLYGON_MODE_LINE } else { wgpu::Features::empty() };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("smoke"),
            required_features: feats,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("device");

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        // Sized so the full help overlay (text + planet image) fits legibly.
        let (w, h) = (1280u32, 720u32);

        // Globals (mirrors Renderer::new).
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("g"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let sky_sh = shader(&device, "sky", include_str!("shaders/sky.wgsl"));
        let terrain_sh = shader(&device, "terrain", include_str!("shaders/terrain.wgsl"));
        let veg_sh = shader(&device, "veg", include_str!("shaders/vegetation.wgsl"));

        let sky_p = make_pipeline(&device, &layout, &sky_sh, &[], format, PassKind::Sky, wgpu::PolygonMode::Fill);
        let terrain_p = make_pipeline(&device, &layout, &terrain_sh, &[vertex_layout()], format, PassKind::Opaque, wgpu::PolygonMode::Fill);
        let veg_p = make_pipeline(&device, &layout, &veg_sh, &[vertex_layout(), instance_layout()], format, PassKind::Opaque, wgpu::PolygonMode::Fill);

        // Overlay pipeline (no bind groups) + its geometry.
        let overlay_sh = shader(&device, "overlay", include_str!("shaders/overlay.wgsl"));
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let overlay_p = make_pipeline(&device, &overlay_layout, &overlay_sh, &[overlay_corner_layout(), overlay_instance_layout()], format, PassKind::Overlay, wgpu::PolygonMode::Fill);
        let overlay_quad = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&OVERLAY_CORNERS),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // Planet image pipeline + texture (validates image.wgsl and shows in the PNG).
        let (planet_view, planet_sampler) = load_planet_texture(&device, &queue);
        let planet_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let planet_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &planet_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&planet_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&planet_sampler) },
            ],
        });
        let image_sh = shader(&device, "image", include_str!("shaders/image.wgsl"));
        let image_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&planet_bgl)],
            immediate_size: 0,
        });
        let image_p = make_pipeline(&device, &image_pl, &image_sh, &[overlay_corner_layout(), overlay_instance_layout()], format, PassKind::Overlay, wgpu::PolygonMode::Fill);

        let (menu_lines, menu_hl) = overlay::menu(&crate::settings::Graphics::default(), 2);
        let overlay_geo = overlay::layout(&menu_lines, menu_hl, w, h);
        assert!(!overlay_geo.quads.is_empty());
        let image_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&overlay_geo.image),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let overlay_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&overlay_geo.quads),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let overlay_count = overlay_geo.quads.len() as u32;

        // One real chunk + base meshes.
        let planet = Planet::new(7);
        let key = ChunkKey { face: 2, level: 8, i: 128, j: 128 };
        let cpu = CpuChunk::build(&planet, key, &mesh::MeshConfig::standard());
        let terrain = GpuMesh::upload(&device, &cpu.vertices, &cpu.indices);
        let trees = InstanceBuf::upload(&device, &cpu.trees);
        let tm = mesh::tree_mesh();
        let tree_mesh = GpuMesh::upload(&device, &tm.vertices, &tm.indices);

        // Camera looking at the chunk from above. Sit well clear of any peak so
        // the eye is never underground.
        let center = key.center_dir() * planet.surface_radius(key.center_dir());
        let eye = center.normalize() * (crate::planet::PLANET_RADIUS + 4000.0);
        let up = crate::planet::tangent_basis(center.normalize()).0;
        let view = Mat4::look_to_rh(eye, (center - eye).normalize(), up);
        let proj = Mat4::perspective_rh(60f32.to_radians(), w as f32 / h as f32, 2.0, 20000.0);
        let vp = proj * view;
        let g = Globals {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 0.0],
            sun_dir: [0.4, 0.7, 0.5, 0.3],
            params: [0.0, crate::planet::PLANET_RADIUS, crate::planet::SEA_LEVEL, 400.0],
            atmosphere: [0.4, 0.6, 0.9, 1.0],
        };
        queue.write_buffer(&globals_buf, 0, bytemuck::bytes_of(&g));

        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = create_depth(&device, w, h);

        // copy_texture_to_buffer requires bytes_per_row to be 256-aligned.
        let unpadded = w * 4;
        let row_bytes = unpadded.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (row_bytes * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &globals_bind, &[]);
            pass.set_pipeline(&sky_p);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&terrain_p);
            pass.set_vertex_buffer(0, terrain.vbuf.slice(..));
            pass.set_index_buffer(terrain.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..terrain.count, 0, 0..1);
            if let Some(t) = &trees {
                pass.set_pipeline(&veg_p);
                draw_instanced(&mut pass, &tree_mesh, t);
            }
            // Overlay on top — validates the overlay shader/pipeline/layout.
            pass.set_pipeline(&overlay_p);
            pass.set_vertex_buffer(0, overlay_quad.slice(..));
            pass.set_vertex_buffer(1, overlay_buf.slice(..));
            pass.draw(0..6, 0..overlay_count);
            // Planet image — validates image.wgsl + texture bind group.
            pass.set_pipeline(&image_p);
            pass.set_bind_group(0, &planet_bind, &[]);
            pass.set_vertex_buffer(0, overlay_quad.slice(..));
            pass.set_vertex_buffer(1, image_buf.slice(..));
            pass.draw(0..6, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "GPU validation error: {err:?}");

        // Read the framebuffer back and confirm real geometry rendered (not a
        // black/uniform screen): the scene must contain both dark sky and a
        // meaningful fraction of brighter, lit terrain/water pixels.
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range();

        // Drop row padding into tight RGBA rows for analysis + PNG.
        let mut rgba = Vec::with_capacity((unpadded * h) as usize);
        for row in 0..h {
            let start = (row * row_bytes) as usize;
            rgba.extend_from_slice(&data[start..start + unpadded as usize]);
        }

        let (mut bright, mut total, mut maxl, mut minl) = (0u32, 0u32, 0.0f32, 1.0f32);
        for px in rgba.chunks_exact(4) {
            let lum = (px[0] as f32 * 0.299 + px[1] as f32 * 0.587 + px[2] as f32 * 0.114) / 255.0;
            maxl = maxl.max(lum);
            minl = minl.min(lum);
            if lum > 0.15 {
                bright += 1;
            }
            total += 1;
        }
        let frac = bright as f32 / total as f32;
        assert!(maxl - minl > 0.1, "frame is nearly uniform (max {maxl:.3} min {minl:.3}) — nothing drew");
        assert!(frac > 0.05, "too few lit pixels ({:.1}%) — terrain likely not visible", frac * 100.0);

        // Dump a PNG so the scene + help overlay can be eyeballed without a window.
        let path = std::env::temp_dir().join("planet_overlay.png");
        let file = std::fs::File::create(&path).expect("create png");
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header().unwrap().write_image_data(&rgba).expect("write png");
        eprintln!("wrote framebuffer to {}", path.display());
    }
}
