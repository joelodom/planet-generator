//! Background music: an embedded **playlist**.
//!
//! The tracks are shuffled into an order, played through start to finish, then
//! reshuffled and played again — forever. A reshuffle never starts on the track
//! that just finished, so you don't hear the same song twice in a row.
//!
//! Adding music is a one-liner: drop an mp3 in `assets/` and add a `(name, bytes)`
//! entry to [`TRACKS`]. Everything is embedded so the app stays self-contained.
//!
//! Audio is best-effort: with no output device the app simply runs silent.

use rand::seq::SliceRandom;
use rand::RngExt;
use std::io::Cursor;

/// The playlist. Order here is irrelevant (it's always shuffled). To add a song,
/// put the mp3 in `assets/` and add a line below.
const TRACKS: &[(&str, &[u8])] = &[
    ("soundtrack", include_bytes!("../assets/soundtrack.mp3")),
    ("Atlas of Dawn", include_bytes!("../assets/soundtrack2.mp3")),
    ("Silver Crown March", include_bytes!("../assets/soundtrack3.mp3")),
    ("Trail in Pine", include_bytes!("../assets/soundtrack4.mp3")),
    ("Trailside Drift", include_bytes!("../assets/soundtrack5.mp3")),
    ("Paper Kite Morning", include_bytes!("../assets/soundtrack6.mp3")),
];

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
        if n > 1 {
            if let Some(last) = self.last_played {
                if order[0] == last {
                    order.swap(0, rng.random_range(1..n));
                }
            }
        }

        let mut queued_last = self.last_played;
        for &i in &order {
            match rodio::Decoder::new(Cursor::new(TRACKS[i].1)) {
                Ok(src) => {
                    self.player.append(src);
                    queued_last = Some(i);
                }
                Err(e) => tracing::warn!(track = TRACKS[i].0, error = %e, "audio: failed to decode track; skipping"),
            }
        }
        self.last_played = queued_last;

        let names: Vec<&str> = order.iter().map(|&i| TRACKS[i].0).collect();
        tracing::info!(order = ?names, "soundtrack playlist shuffled");
    }
}

#[cfg(test)]
mod tests {
    use super::TRACKS;
    use rodio::Source;

    #[test]
    fn all_tracks_decode() {
        assert!(TRACKS.len() >= 2, "playlist should have multiple tracks");
        for (name, bytes) in TRACKS {
            let d = rodio::Decoder::new(std::io::Cursor::new(*bytes))
                .unwrap_or_else(|e| panic!("decode {name}: {e}"));
            assert!(d.sample_rate().get() > 0, "{name} has no sample rate");
            assert!(d.channels().get() >= 1, "{name} has no channels");
        }
    }
}
