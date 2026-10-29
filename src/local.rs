//! `--local`: transcription with the local MLX Voxtral model, no API call.
//! Only compiled with the `local` feature.

use std::collections::HashSet;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender};

use voxtral_mlx::audio::LiveResampler;
use voxtral_mlx::model::encode::StreamEncoder;
use voxtral_mlx::model::session::StreamSession;
use voxtral_mlx::tokenizer::TekkenEncoder;
use voxtral_mlx::{audio, loader, memory};

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

    guard_memory();
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

    let (pairs, tail) = rest.as_chunks::<2>();
    for &[lo, hi] in pairs {
        samples.push(sample(lo, hi));
    }
    if let [last] = tail {
        *odd = Some(*last);
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
    Ok(ids
        .into_iter()
        .map(|id| (id, CONTEXT_BIAS_STRENGTH))
        .collect())
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
    guard_memory();
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

    guard_memory();
    let mut encoder = StreamEncoder::new(&mut model);
    // One second per feed, the order of magnitude a capture delivers.
    for chunk in resampled.samples.chunks(16_000) {
        encoder.feed(&mut model, chunk)?;
    }

    let embeds = encoder.finish(&mut model)?.ok_or("No audio")?;
    let tokens = model.transcribe_tokens_from_embeds(&embeds, 0.0, &logit_bias)?;
    Ok(tokenizer.decode(&tokens)?.trim().to_string())
}

/// Continuous captions: prints text as it is decoded, for as long as samples
/// arrive. The caller drops the sender to stop. Returns the whole transcript.
///
/// This uses the streaming decoder, which emits text before it has heard the
/// end of a sentence. It reads worse than the batch decoder, and it never
/// revises what it already printed. Custom word bias does not apply.
pub fn caption(
    samples: Receiver<Vec<f32>>,
    input_rate: u32,
    weights: &str,
) -> std::result::Result<String, String> {
    captions(samples, input_rate, weights).map_err(|e| e.to_string())
}

/// The encoder KV cache grows by concatenation on every step and only stops
/// at max_size=100_000, which no session reaches. Left alone it costs about
/// 13 MB per second of audio, and every step copies the whole cache, so the
/// allocator holds one buffer of every size it has seen. A session is
/// therefore closed and reopened on this window.
const WINDOW_SECONDS: usize = 30;

fn captions(samples: Receiver<Vec<f32>>, input_rate: u32, weights: &str) -> Result<String> {
    let snapshot = loader::resolve_weights_dir(weights)?;
    let (mut model, tokenizer) =
        loader::load_model(snapshot.to_str().ok_or("non-utf8 weights path")?)?;

    guard_memory();

    let mut resampler = LiveResampler::new(input_rate, 16_000)?;
    let mut session = StreamSession::new(&mut model, 0.0)?;
    let mut tokens: Vec<u32> = Vec::new();
    // Text of the open session, against the whole transcript.
    let mut window = String::new();
    let mut transcript = String::new();
    let mut fed = 0usize;
    let window_samples = WINDOW_SECONDS * 16_000;

    while let Ok(chunk) = samples.recv() {
        let resampled = resampler.process(&chunk)?;
        if resampled.is_empty() {
            continue;
        }
        fed += resampled.len();

        let new = session.feed(&mut model, &resampled)?;
        if !new.is_empty() {
            tokens.extend(new);
            emit(&tokenizer.decode(&tokens)?, &mut window, &mut transcript);
        }

        if fed >= window_samples {
            tokens.extend(session.flush(&mut model)?);
            emit(&tokenizer.decode(&tokens)?, &mut window, &mut transcript);
            session = StreamSession::new(&mut model, 0.0)?;
            tokens.clear();
            window.clear();
            fed = 0;
        }
    }

    tokens.extend(session.flush(&mut model)?);
    emit(&tokenizer.decode(&tokens)?, &mut window, &mut transcript);
    println!();
    Ok(transcript)
}

/// Prints whatever the decoder added since the last call. Tokens only ever
/// append, so a full decode extends the previous one.
fn emit(text: &str, window: &mut String, transcript: &mut String) {
    let Some(added) = text.strip_prefix(window.as_str()) else {
        return;
    };
    if added.is_empty() {
        return;
    }
    print!("{added}");
    io::stdout().flush().ok();
    window.push_str(added);
    transcript.push_str(added);
}

/// Sends a WAV file's samples at 16 kHz, in one second chunks, the order of
/// magnitude a capture delivers.
pub fn send_file(wav_data: &[u8], sender: &SyncSender<Vec<f32>>) -> Result<()> {
    let raw = audio::load_wav_bytes(wav_data)?;
    let resampled = audio::resample_to_16k(&raw)?;
    for chunk in resampled.samples.chunks(16_000) {
        if sender.send(chunk.to_vec()).is_err() {
            break;
        }
    }
    Ok(())
}

/// Sends 16 kHz mono 16-bit PCM read from `input`.
pub fn send_pcm(mut input: impl Read, sender: &SyncSender<Vec<f32>>) -> Result<()> {
    let mut buffer = [0u8; 8192];
    let mut odd: Option<u8> = None;
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            return Ok(());
        }
        let samples = to_samples(&buffer[..read], &mut odd);
        if !samples.is_empty() && sender.send(samples).is_err() {
            return Ok(());
        }
    }
}

/// Bytes the allocator may hold for reuse between steps.
const CACHE_LIMIT: usize = 512 * 1024 * 1024;

/// Stops the process before the machine runs out of memory. MLX allocates
/// through Metal, so this memory never shows in the resident set size and a
/// runaway session freezes the machine with no warning from `ps`.
///
/// The encoder cache grows with the length of the take, so a long one walks
/// into the ceiling. 7 GB leaves more than 2 GB over the measured peak.
/// Override it with REC_MEMORY_LIMIT_GB.
fn guard_memory() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(guard_memory_once);
}

fn guard_memory_once() {
    memory::set_cache_limit(CACHE_LIMIT).ok();

    let limit_gb: usize = std::env::var("REC_MEMORY_LIMIT_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let limit = limit_gb * 1024 * 1024 * 1024;
    let debug = std::env::var("REC_MEMORY_DEBUG").is_ok();

    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let active = memory::active_memory().unwrap_or(0);
            let cached = memory::cache_memory().unwrap_or(0);
            let used = active + cached;
            if debug {
                eprintln!(
                    "rec: mlx active {} MB, cache {} MB",
                    active / (1024 * 1024),
                    cached / (1024 * 1024)
                );
            }
            if used > limit {
                eprintln!(
                    "\nrec: MLX is holding {} GB, over the {} GB ceiling. Stopping.",
                    used / (1024 * 1024 * 1024),
                    limit_gb
                );
                std::process::exit(70);
            }
        }
    });
}
