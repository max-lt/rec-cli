//! `--local`: transcription with the local MLX Voxtral model, no API call.
//! Only compiled with the `local` feature.

use std::collections::HashSet;
use std::io::Read;
use std::path::Path;
use std::sync::mpsc::Receiver;

use voxtral_mlx::model::encode::StreamEncoder;
use voxtral_mlx::tokenizer::TekkenEncoder;
use voxtral_mlx::audio::LiveResampler;
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
    let logit_bias = logit_bias(&snapshot, bias_words)?;

    let mel = loader::mel_array(audio::log_mel_frames(wav_data)?)?;
    let tokens = model.transcribe_tokens_biased(&mel, 0.0, &logit_bias)?;
    Ok(tokenizer.decode(&tokens)?.trim().to_string())
}

/// Reads 16 kHz mono 16-bit PCM from `input` and runs the encoder while the
/// samples arrive, so only the decoder is left when the capture ends.
pub fn transcribe_stream(
    mut input: impl Read,
    weights: &str,
    bias_words: &[String],
) -> Result<String> {
    let snapshot = loader::resolve_weights_dir(weights)?;
    let (mut model, tokenizer) =
        loader::load_model(snapshot.to_str().ok_or("non-utf8 weights path")?)?;
    let logit_bias = logit_bias(&snapshot, bias_words)?;

    let mut encoder = StreamEncoder::new(&mut model);
    let mut buffer = [0u8; 8192];
    // A read can split a sample in half.
    let mut odd: Option<u8> = None;

    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let samples = to_samples(&buffer[..read], &mut odd);
        if !samples.is_empty() {
            encoder.feed(&mut model, &samples)?;
        }
    }

    let embeds = encoder.finish(&mut model)?.ok_or("No audio")?;
    let tokens = model.transcribe_tokens_from_embeds(&embeds, 0.0, &logit_bias)?;
    Ok(tokenizer.decode(&tokens)?.trim().to_string())
}

/// Little-endian 16-bit samples to f32, carrying a split sample across reads.
fn to_samples(bytes: &[u8], odd: &mut Option<u8>) -> Vec<f32> {
    let mut samples = Vec::with_capacity(bytes.len() / 2 + 1);
    let mut rest = bytes;

    if let Some(first) = odd.take() {
        if let Some((second, tail)) = rest.split_first() {
            samples.push(sample(first, *second));
            rest = tail;
        } else {
            *odd = Some(first);
            return samples;
        }
    }

    let pairs = rest.len() / 2;
    for pair in rest[..pairs * 2].chunks_exact(2) {
        samples.push(sample(pair[0], pair[1]));
    }
    if rest.len() % 2 == 1 {
        *odd = Some(rest[rest.len() - 1]);
    }
    samples
}

fn sample(low: u8, high: u8) -> f32 {
    i16::from_le_bytes([low, high]) as f32 / 32768.0
}

/// Words get boosted both standalone and with a leading space (how they
/// appear mid-sentence after tekken BPE).
fn logit_bias(snapshot: &Path, bias_words: &[String]) -> Result<Vec<(u32, f32)>> {
    if bias_words.is_empty() {
        return Ok(vec![]);
    }
    let encoder = TekkenEncoder::from_file(snapshot.join("tekken.json"))?;
    let mut ids = HashSet::new();
    for word in bias_words {
        ids.extend(encoder.encode(word));
        ids.extend(encoder.encode(&format!(" {word}")));
    }
    Ok(ids.into_iter().map(|id| (id, CONTEXT_BIAS_STRENGTH)).collect())
}

/// Transcribes a live capture: the caller sends mono samples at `input_rate`
/// and drops the sender to stop. The encoder runs while the samples arrive,
/// so only the decoder is left when the recording ends.
///
/// MLX must stay on one thread, so the model lives here and never leaves it.
pub fn transcribe_live(
    samples: Receiver<Vec<f32>>,
    input_rate: u32,
    weights: &str,
    bias_words: &[String],
) -> std::result::Result<String, String> {
    live(samples, input_rate, weights, bias_words).map_err(|e| e.to_string())
}

fn live(
    samples: Receiver<Vec<f32>>,
    input_rate: u32,
    weights: &str,
    bias_words: &[String],
) -> Result<String> {
    let snapshot = loader::resolve_weights_dir(weights)?;
    let (mut model, tokenizer) =
        loader::load_model(snapshot.to_str().ok_or("non-utf8 weights path")?)?;
    let logit_bias = logit_bias(&snapshot, bias_words)?;

    let mut resampler = LiveResampler::new(input_rate, 16_000)?;
    let mut encoder = StreamEncoder::new(&mut model);

    while let Ok(chunk) = samples.recv() {
        let resampled = resampler.process(&chunk)?;
        if !resampled.is_empty() {
            encoder.feed(&mut model, &resampled)?;
        }
    }

    let embeds = encoder.finish(&mut model)?.ok_or("No audio")?;
    let tokens = model.transcribe_tokens_from_embeds(&embeds, 0.0, &logit_bias)?;
    Ok(tokenizer.decode(&tokens)?.trim().to_string())
}

/// Same encoder path as a live capture, over a file already on disk. It saves
/// no time, since nothing overlaps, but it transcribes a fixed take the way a
/// capture would, which is how the two paths get compared on equal audio.
pub fn transcribe_chunked(wav_data: &[u8], weights: &str, bias_words: &[String]) -> Result<String> {
    let snapshot = loader::resolve_weights_dir(weights)?;
    let (mut model, tokenizer) =
        loader::load_model(snapshot.to_str().ok_or("non-utf8 weights path")?)?;
    let logit_bias = logit_bias(&snapshot, bias_words)?;

    let raw = audio::load_wav_bytes(wav_data)?;
    let resampled = audio::resample_to_16k(&raw)?;

    let mut encoder = StreamEncoder::new(&mut model);
    // One second per feed, the order of magnitude a capture delivers.
    for chunk in resampled.samples.chunks(16_000) {
        encoder.feed(&mut model, chunk)?;
    }

    let embeds = encoder.finish(&mut model)?.ok_or("No audio")?;
    let tokens = model.transcribe_tokens_from_embeds(&embeds, 0.0, &logit_bias)?;
    Ok(tokenizer.decode(&tokens)?.trim().to_string())
}
