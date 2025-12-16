//! rec - Quick speech-to-text for devs

use arboard::Clipboard;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use reqwest::multipart;
use serde::Deserialize;
use std::io::{self, BufWriter, Write};
use std::sync::{Arc, Mutex};

const API_URL: &str = "https://api.mistral.ai/v1/audio/transcriptions";
const MODEL: &str = "voxtral-mini-2507";

#[derive(Parser)]
#[command(name = "rec", about = "Quick speech-to-text for devs")]
struct Args {
    /// Copy result to clipboard
    #[arg(short, long)]
    clip: bool,
}

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
}

/// Clear line and print status
fn status(msg: &str) {
    eprint!("\r\x1b[K{}", msg);
    io::stderr().flush().ok();
}

/// Move up one line, clear it, and print status
fn status_up(msg: &str) {
    eprint!("\x1b[A\r\x1b[K{}", msg);
    io::stderr().flush().ok();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    #[cfg(debug_assertions)]
    dotenvy::dotenv().ok();

    let api_key = std::env::var("MISTRAL_API_KEY").map_err(|_| "MISTRAL_API_KEY not set")?;

    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("No mic")?;
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();

    status("Recording...");

    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_clone = samples.clone();

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                samples_clone.lock().unwrap().extend_from_slice(data);
            },
            |err| eprintln!("Error: {}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _: &_| {
                let floats: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                samples_clone.lock().unwrap().extend(floats);
            },
            |err| eprintln!("Error: {}", err),
            None,
        )?,
        _ => return Err("Unsupported format".into()),
    };

    stream.play()?;

    // Wait for Enter
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    drop(stream);

    let recorded = samples.lock().unwrap();
    let duration = recorded.len() as f32 / sample_rate as f32 / channels as f32;

    if recorded.is_empty() {
        status_up("No audio\n");
        return Err("No audio".into());
    }

    status_up(&format!("{:.1}s transcribing...", duration));

    // Encode WAV
    let mut wav_buffer = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut wav_buffer);
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = WavWriter::new(BufWriter::new(cursor), spec)?;
        for &s in recorded.iter() {
            writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
        }
        writer.finalize()?;
    }

    // Transcribe
    let client = reqwest::Client::new();
    let form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(wav_buffer)
                .file_name("audio.wav")
                .mime_str("audio/wav")?,
        )
        .text("model", MODEL);

    let resp = client
        .post(API_URL)
        .header("x-api-key", &api_key)
        .multipart(form)
        .send()
        .await?;

    if !resp.status().is_success() {
        let body = resp.text().await?;
        status(&format!("API error: {}\n", body));
        return Err(format!("API error: {}", body).into());
    }

    let result: TranscriptionResponse = resp.json().await?;
    status("");
    println!("{}", result.text);

    if args.clip {
        Clipboard::new()?.set_text(&result.text)?;
    }

    Ok(())
}
