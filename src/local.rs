//! `--local`: transcription with the local MLX Voxtral model, no API call.
//! Only compiled with the `local` feature.

use std::collections::HashSet;

use voxtral_mlx::tokenizer::TekkenEncoder;
use voxtral_mlx::{audio, loader};

pub const DEFAULT_WEIGHTS: &str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit";

/// Additive logit boost for --bias vocabulary words: enough to tip the
/// balance toward a known word the model already finds plausible, without
/// forcing it into unrelated audio.
const CONTEXT_BIAS_STRENGTH: f32 = 3.0;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn transcribe(wav_data: &[u8], weights: &str, bias_words: &[String]) -> Result<String> {
    let snapshot = loader::resolve_weights_dir(weights)?;
    let (mut model, tokenizer) =
        loader::load_model(snapshot.to_str().ok_or("non-utf8 weights path")?)?;

    // Words get boosted both standalone and with a leading space (how they
    // appear mid-sentence after tekken BPE).
    let logit_bias: Vec<(u32, f32)> = if bias_words.is_empty() {
        vec![]
    } else {
        let encoder = TekkenEncoder::from_file(snapshot.join("tekken.json"))?;
        let mut ids = HashSet::new();
        for word in bias_words {
            ids.extend(encoder.encode(word));
            ids.extend(encoder.encode(&format!(" {word}")));
        }
        ids.into_iter()
            .map(|id| (id, CONTEXT_BIAS_STRENGTH))
            .collect()
    };

    let mel = loader::mel_array(audio::log_mel_frames(wav_data)?)?;
    let tokens = model.transcribe_tokens_biased(&mel, 0.0, &logit_bias)?;
    Ok(tokenizer.decode(&tokens)?.trim().to_string())
}
