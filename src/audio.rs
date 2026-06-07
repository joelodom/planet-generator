//! Background soundtrack. The track is embedded in the binary and looped for the
//! entire lifetime of the app — keeping the returned [`Audio`] alive keeps it
//! playing; dropping it stops the music. Audio failures are non-fatal: if there
//! is no output device, the app simply runs silent.

use std::io::Cursor;

/// The soundtrack, baked into the binary so the app is self-contained.
const SOUNDTRACK: &[u8] = include_bytes!("../assets/soundtrack.mp3");

/// Holds the audio output alive. Drop to stop playback.
pub struct Audio {
    _stream: rodio::MixerDeviceSink,
    _player: rodio::Player,
}

impl Audio {
    /// Start looping the soundtrack. Returns `None` (and logs) if audio can't be
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
        match rodio::Decoder::new_looped(Cursor::new(SOUNDTRACK)) {
            Ok(source) => {
                player.append(source);
                player.set_volume(volume);
                tracing::info!(volume, bytes = SOUNDTRACK.len(), "soundtrack started (looping)");
                Some(Self { _stream: stream, _player: player })
            }
            Err(e) => {
                tracing::warn!(error = %e, "audio: failed to decode soundtrack; running silent");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SOUNDTRACK;
    use rodio::Source;

    #[test]
    fn embedded_soundtrack_decodes() {
        // The decode path is the part we can verify without an audio device.
        let decoder = rodio::Decoder::new(std::io::Cursor::new(SOUNDTRACK)).expect("decode soundtrack mp3");
        assert!(decoder.sample_rate().get() > 0, "soundtrack has no sample rate");
        assert!(decoder.channels().get() >= 1, "soundtrack has no channels");
        // And it loops without error.
        let _looped = rodio::Decoder::new_looped(std::io::Cursor::new(SOUNDTRACK)).expect("loop soundtrack");
    }
}
