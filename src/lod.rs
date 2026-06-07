//! Level-of-detail streaming: a quadtree over each of the six cube-sphere faces,
//! plus a background thread pool that meshes chunks off the main thread.
//!
//! The main thread never generates geometry inline (except the six root chunks
//! at startup, so there is always *something* to draw). Each frame it walks the
//! quadtree, decides which nodes should be visible, draws the ones already on
//! the GPU, and asks the worker pool for the ones it's missing — falling back to
//! a coarser ancestor in the meantime so there are never holes.

use crate::mesh::CpuChunk;
use crate::planet::{self, Planet, FACES, PLANET_RADIUS};
use glam::Vec3;
use std::collections::{HashSet, VecDeque};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Deepest quadtree subdivision. Level 0 is one chunk per cube face; each level
/// quarters the area. Level 9 gives sub-metre-ish detail near the ground.
pub const MAX_LEVEL: u32 = 9;

/// How eagerly chunks subdivide. A node splits when the camera is within
/// `SPLIT_FACTOR * node_world_size` of it. Higher = more detail, more chunks.
pub const SPLIT_FACTOR: f32 = 1.7;

/// Identifies one quadtree node: which cube face, how deep, and where within the
/// face's index grid. Small and `Copy`, so it's the key for every cache/set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkKey {
    pub face: u8,
    pub level: u32,
    pub i: u32,
    pub j: u32,
}

impl ChunkKey {
    /// The six top-level chunks, one per cube face.
    pub fn roots() -> [ChunkKey; 6] {
        std::array::from_fn(|f| ChunkKey { face: f as u8, level: 0, i: 0, j: 0 })
    }

    /// This node's square within face space: origin (u0, v0) and edge `size`,
    /// all in cube coordinates where a full face spans [-1, 1].
    pub fn face_rect(&self) -> (f32, f32, f32) {
        let size = 2.0 / (1u32 << self.level) as f32;
        (-1.0 + self.i as f32 * size, -1.0 + self.j as f32 * size, size)
    }

    /// Unit direction from the planet centre to this node's centre.
    pub fn center_dir(&self) -> Vec3 {
        let (u0, v0, size) = self.face_rect();
        let face = &FACES[self.face as usize];
        let cube = face.base + face.right * (u0 + size * 0.5) + face.up * (v0 + size * 0.5);
        planet::cube_to_sphere(cube)
    }

    /// Approximate world-space edge length of this chunk.
    pub fn world_size(&self) -> f32 {
        let size = 2.0 / (1u32 << self.level) as f32;
        size * PLANET_RADIUS
    }

    pub fn children(&self) -> [ChunkKey; 4] {
        let (l, i, j) = (self.level + 1, self.i * 2, self.j * 2);
        [
            ChunkKey { face: self.face, level: l, i, j },
            ChunkKey { face: self.face, level: l, i: i + 1, j },
            ChunkKey { face: self.face, level: l, i, j: j + 1 },
            ChunkKey { face: self.face, level: l, i: i + 1, j: j + 1 },
        ]
    }

    /// Stable per-chunk seed mixing the planet seed with the coordinates, so
    /// vegetation placement is reproducible.
    pub fn hash(&self, seed: u64) -> u64 {
        let mut h = seed;
        for v in [self.face as u64, self.level as u64, self.i as u64, self.j as u64] {
            h ^= v.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(h << 6).wrapping_add(h >> 2);
            h = h.wrapping_mul(0x100_0000_01B3);
        }
        h
    }
}

/// Result of one frame's quadtree walk.
pub struct Selection {
    /// Chunks that are ready on the GPU and should be drawn this frame.
    pub draw: Vec<ChunkKey>,
    /// Chunks we want but don't have yet — to be requested from the worker pool.
    pub want: Vec<ChunkKey>,
}

/// Walk all six faces and decide what to draw / request, given a predicate that
/// reports which chunks are already resident on the GPU.
pub fn select(planet: &Planet, cam: Vec3, ready: &dyn Fn(ChunkKey) -> bool) -> Selection {
    let mut sel = Selection { draw: Vec::new(), want: Vec::new() };
    let cam_len = cam.length();
    // Angular radius of the horizon cone as seen from the camera. Anything more
    // than this far around the sphere is occluded by the planet itself.
    let horizon = if cam_len > PLANET_RADIUS { (PLANET_RADIUS / cam_len).acos() } else { std::f32::consts::PI };
    for root in ChunkKey::roots() {
        select_node(root, planet, cam, horizon, ready, &mut sel);
    }
    sel
}

fn select_node(node: ChunkKey, planet: &Planet, cam: Vec3, horizon: f32, ready: &dyn Fn(ChunkKey) -> bool, sel: &mut Selection) {
    // Horizon cull: skip nodes fully behind the planet's bulge.
    let center_dir = node.center_dir();
    let ang = center_dir.angle_between(cam.normalize_or_zero());
    let node_ang = node.world_size() / PLANET_RADIUS; // ~angular radius
    if ang > horizon + node_ang + 0.06 {
        return;
    }

    let center = center_dir * planet.surface_radius(center_dir);
    let dist = (center - cam).length();
    let split = dist < SPLIT_FACTOR * node.world_size() && node.level < MAX_LEVEL;

    if split {
        let kids = node.children();
        if kids.iter().all(|k| ready(*k)) {
            for k in kids {
                select_node(k, planet, cam, horizon, ready, sel);
            }
            return;
        }
        // Children not all ready: request the missing ones, draw the parent.
        for k in kids {
            if !ready(k) {
                sel.want.push(k);
            }
        }
    }

    if ready(node) {
        sel.draw.push(node);
    } else {
        sel.want.push(node);
    }
}

// ---------------------------------------------------------------------------
// Background meshing pool
// ---------------------------------------------------------------------------

struct Queue {
    inner: Mutex<QueueInner>,
    cv: Condvar,
}

struct QueueInner {
    jobs: VecDeque<ChunkKey>,
    queued: HashSet<ChunkKey>,
    shutdown: bool,
}

/// Owns the worker threads and tracks which chunks are in flight.
pub struct Streamer {
    queue: Arc<Queue>,
    results: Receiver<(ChunkKey, CpuChunk)>,
    pending: HashSet<ChunkKey>,
    handles: Vec<JoinHandle<()>>,
}

impl Streamer {
    pub fn new(planet: Arc<Planet>, threads: usize) -> Self {
        let queue = Arc::new(Queue {
            inner: Mutex::new(QueueInner { jobs: VecDeque::new(), queued: HashSet::new(), shutdown: false }),
            cv: Condvar::new(),
        });
        let (tx, results) = channel();
        let mut handles = Vec::new();
        let n = threads.max(1);
        for i in 0..n {
            handles.push(spawn_worker(i, queue.clone(), planet.clone(), tx.clone()));
        }
        tracing::info!(workers = n, "chunk meshing pool started");
        Self { queue, results, pending: HashSet::new(), handles }
    }

    /// Queue a chunk for meshing if it isn't already in flight.
    pub fn request(&mut self, key: ChunkKey) {
        if self.pending.contains(&key) {
            return;
        }
        let mut inner = self.queue.inner.lock().unwrap();
        if inner.queued.insert(key) {
            inner.jobs.push_back(key);
            self.pending.insert(key);
            self.queue.cv.notify_one();
        }
    }

    /// Drain finished chunks. Returned chunks have left the in-flight set.
    pub fn poll(&mut self) -> Vec<(ChunkKey, CpuChunk)> {
        let mut out = Vec::new();
        while let Ok((key, chunk)) = self.results.try_recv() {
            self.pending.remove(&key);
            out.push((key, chunk));
        }
        out
    }

    #[allow(dead_code)]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        {
            let mut inner = self.queue.inner.lock().unwrap();
            inner.shutdown = true;
        }
        self.queue.cv.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn spawn_worker(id: usize, queue: Arc<Queue>, planet: Arc<Planet>, tx: Sender<(ChunkKey, CpuChunk)>) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("chunk-worker-{id}"))
        .spawn(move || loop {
            let key = {
                let mut inner = queue.inner.lock().unwrap();
                loop {
                    if inner.shutdown {
                        return;
                    }
                    if let Some(k) = inner.jobs.pop_front() {
                        inner.queued.remove(&k);
                        break k;
                    }
                    inner = queue.cv.wait(inner).unwrap();
                }
            };
            let chunk = CpuChunk::build(&planet, key);
            if tx.send((key, chunk)).is_err() {
                return; // main thread gone
            }
        })
        .expect("spawn chunk worker")
}
