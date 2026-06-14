//! Background music: an embedded **playlist**.
//!
//! The tracks are shuffled into an order, played through start to finish, then
//! reshuffled and played again — forever. A reshuffle never starts on the track
//! that just finished, so you don't hear the same song twice in a row.
//!
//! Adding music is a one-liner: drop an mp3 in `assets/` and add an `include_bytes!`
//! line to [`TRACKS`]. Everything is embedded so the app stays self-contained.
//!
//! Audio is best-effort: with no output device the app simply runs silent.

use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::io::Cursor;

/// The playlist. Order here is irrelevant (it's always shuffled). To add a song,
/// put the mp3 in `assets/` and add a line below.
const TRACKS: &[&[u8]] = &[
    include_bytes!("../assets/soundtrack.mp3"),
    include_bytes!("../assets/soundtrack2.mp3"),
    include_bytes!("../assets/soundtrack3.mp3"),
    include_bytes!("../assets/soundtrack4.mp3"),
    include_bytes!("../assets/soundtrack5.mp3"),
    include_bytes!("../assets/soundtrack6.mp3"),
    include_bytes!("../assets/soundtrack7.mp3"),
    include_bytes!("../assets/soundtrack8.mp3"),
    include_bytes!("../assets/soundtrack9.mp3"),
];

/// A shuffled play order of the embedded soundtrack, returned as raw mp3 bytes —
/// the same "shuffle the playlist, play it through" idea the live player uses
/// ([`Audio::enqueue_shuffled`]), but **seeded** so a given planet seed always
/// yields the same soundtrack (matches the repo's derive-everything-from-seed
/// rule). The headless `--video` recorder muxes these in as background music.
pub fn shuffled_soundtrack(seed: u64) -> Vec<&'static [u8]> {
    let mut order: Vec<usize> = (0..TRACKS.len()).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    order.shuffle(&mut rng);
    order.into_iter().map(|i| TRACKS[i]).collect()
}

/// Holds the audio output alive and drives the playlist. Drop to stop playback.
pub struct Audio {
    _stream: rodio::MixerDeviceSink,
    player: rodio::Player,
    /// Index of the track queued last, so the next shuffle won't start on it.
    last_played: Option<usize>,
}

impl Audio {
    /// Start the playlist. Returns `None` (and logs) if audio can't be
    /// initialised, so the caller can carry on without sound.
    pub fn start(volume: f32) -> Option<Self> {
        let stream = match rodio::DeviceSinkBuilder::open_default_sink() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "audio: no output device; running silent");
                return None;
            }
        };
        let player = rodio::Player::connect_new(stream.mixer());
        player.set_volume(volume);
        let mut audio = Self { _stream: stream, player, last_played: None };
        audio.enqueue_shuffled();
        tracing::info!(volume, tracks = TRACKS.len(), "soundtrack playlist started");
        Some(audio)
    }

    /// Call once per frame. When the queued playlist has finished, reshuffle and
    /// queue the next round. Cheap (one atomic check) when music is still playing.
    pub fn tick(&mut self) {
        if self.player.empty() {
            self.enqueue_shuffled();
        }
    }

    /// Build a fresh shuffled order (never starting on the last track played) and
    /// queue every track in it back-to-back.
    fn enqueue_shuffled(&mut self) {
        let n = TRACKS.len();
        if n == 0 {
            return;
        }
        let mut rng = rand::rng();
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        // Don't replay the just-finished track immediately.
        if n > 1 && let Some(last) = self.last_played && order[0] == last {
            order.swap(0, rng.random_range(1..n));
        }

        let mut queued_last = self.last_played;
        for &i in &order {
            match rodio::Decoder::new(Cursor::new(TRACKS[i])) {
                Ok(src) => {
                    self.player.append(src);
                    queued_last = Some(i);
                }
                Err(e) => tracing::warn!(track = i, error = %e, "audio: failed to decode track; skipping"),
            }
        }
        self.last_played = queued_last;

        tracing::info!(order = ?order, "soundtrack playlist shuffled");
    }
}

#[cfg(test)]
mod tests {
    use super::{shuffled_soundtrack, TRACKS};
    use rodio::Source;

    #[test]
    fn shuffled_soundtrack_is_seed_deterministic() {
        // Same seed → identical play order (the property the video recorder relies on);
        // different seeds generally differ. Compare by track bytes' identity (ptr).
        let key = |v: &[&[u8]]| v.iter().map(|b| b.as_ptr() as usize).collect::<Vec<_>>();
        assert_eq!(key(&shuffled_soundtrack(7)), key(&shuffled_soundtrack(7)));
        assert_ne!(key(&shuffled_soundtrack(1)), key(&shuffled_soundtrack(2)));
        // A shuffle is a permutation: every track appears exactly once.
        assert_eq!(shuffled_soundtrack(7).len(), TRACKS.len());
    }

    #[test]
    fn all_tracks_decode() {
        assert!(TRACKS.len() >= 2, "playlist should have multiple tracks");
        for (i, bytes) in TRACKS.iter().enumerate() {
            let d = rodio::Decoder::new(std::io::Cursor::new(*bytes))
                .unwrap_or_else(|e| panic!("decode track {i}: {e}"));
            assert!(d.sample_rate().get() > 0, "track {i} has no sample rate");
            assert!(d.channels().get() >= 1, "track {i} has no channels");
        }
    }
}
