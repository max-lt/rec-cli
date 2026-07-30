//! `--local`: transcription with the local MLX Voxtral model, no API call.
//! Only compiled with the `local` feature.

use std::path::{Path, PathBuf};

use voxtral_mlx::audio::{self, mel::MelSpectrogram, pad::PadConfig};
use voxtral_mlx::model::config::VoxtralConfig;
use voxtral_mlx::model::generate::VoxtralRealtime;
use voxtral_mlx::model::weights::WeightMap;
use voxtral_mlx::tokenizer::VoxtralTokenizer;

pub const DEFAULT_WEIGHTS: &str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit";

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// `weights` is either a local HF snapshot directory or a HuggingFace repo
/// id, resolved through ~/.cache/huggingface (shared with voxtral-mlx and
/// any Python voxmlx install; downloads on first use).
fn resolve_weights_dir(weights: &str) -> Result<PathBuf> {
    let path = Path::new(weights);
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(weights.to_string());
    let config_path = repo.get("config.json")?;
    repo.get("model.safetensors")?;
    repo.get("tekken.json")?;

    Ok(config_path
        .parent()
        .ok_or("resolve snapshot directory")?
        .to_path_buf())
}

pub fn transcribe(wav_data: &[u8], weights: &str) -> Result<String> {
    let snapshot = resolve_weights_dir(weights)?;

    let config = VoxtralConfig::from_file(snapshot.join("config.json"))?;
    let quant = config
        .quantization
        .as_ref()
        .ok_or("config.json missing quantization block")?;
    let mut weight_map = WeightMap::load(
        snapshot.join("model.safetensors"),
        quant.group_size,
        quant.bits,
    )?;
    let mut model = VoxtralRealtime::load(&mut weight_map, &config)?;
    let tokenizer = VoxtralTokenizer::from_file(snapshot.join("tekken.json"))?;

    let raw = audio::load_wav_bytes(wav_data)?;
    let resampled = audio::resample_to_16k(&raw)?;
    let padded = audio::pad_audio(&resampled, &PadConfig::voxtral());

    let mel_frames = MelSpectrogram::voxtral().compute_log(&padded.samples); // [T, n_mels]
    let t = mel_frames.len() as i32;
    let n_mels = mel_frames.first().ok_or("empty audio")?.len() as i32;
    let flat: Vec<f32> = mel_frames.into_iter().flatten().collect();
    let mel = mlx_rs::Array::from_slice(&flat, &[t, n_mels]).transpose()?; // [n_mels, T]

    let tokens = model.transcribe_tokens(&mel, 0.0)?;
    Ok(tokenizer.decode(&tokens)?.trim().to_string())
}
