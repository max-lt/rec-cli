//! rec - Quick speech-to-text for devs

mod config;
mod correction;

use arboard::Clipboard;
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use reqwest::multipart;
use serde::Deserialize;
use std::io::{self, BufWriter, Write};
use std::sync::{Arc, Mutex};

const API_URL: &str = "https://api.mistral.ai/v1/audio/transcriptions";
const MODEL_V1: &str = "voxtral-mini-2507";
const MODEL_V2: &str = "voxtral-mini-2602";

#[derive(Parser)]
#[command(name = "rec", about = "Quick speech-to-text for devs")]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Audio file to transcribe (instead of recording)
    #[arg(short, long, global = true)]
    file: Option<std::path::PathBuf>,

    /// Copy result to clipboard
    #[arg(short, long, global = true)]
    clip: bool,

    /// Correct transcription using Claude API
    #[arg(long, global = true)]
    correct: bool,

    /// Show Claude's correction comments
    #[arg(long, global = true)]
    debug: bool,

    /// Use voxtral-mini-2602 model (v2)
    #[arg(long, global = true)]
    v2: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a custom word to the vocabulary (for Claude correction)
    AddWord { word: String },
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

    // Handle add-word subcommand
    if let Some(Commands::AddWord { word }) = args.command {
        let mut config = config::Config::load()?;
        config.add_custom_word(word.clone());
        config.save()?;
        eprintln!("Word added: {}", word);
        return Ok(());
    }

    // Transcribe command (default behavior)
    let api_key = std::env::var("MISTRAL_API_KEY").map_err(|_| "MISTRAL_API_KEY not set")?;

    let wav_buffer = if let Some(path) = &args.file {
        // Read audio file
        status("Reading file...");
        std::fs::read(path)?
    } else {
        // Record from microphone
        status("Loading...");

        let host = cpal::default_host();
        let device = host.default_input_device().ok_or("No mic")?;
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate();
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
        wav_buffer
    };

    status("Transcribing...");

    // Transcribe
    let model = if args.v2 { MODEL_V2 } else { MODEL_V1 };
    let client = reqwest::Client::new();
    let form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(wav_buffer)
                .file_name("audio.wav")
                .mime_str("audio/wav")?,
        )
        .text("model", model);

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

    let final_text = if args.correct {
        status("Correcting with Claude...");

        let anthropic_key =
            std::env::var("ANTHROPIC_API_KEY").map_err(|_| "ANTHROPIC_API_KEY not set")?;

        let config = config::Config::load()?;
        let history = config::Config::load_history().unwrap_or_default();

        match correction::correct_transcription(
            &result.text,
            &config.custom_words,
            &config.claude_model,
            &anthropic_key,
            &history,
        )
        .await
        {
            Ok(output) => {
                status("");

                // Check if correction was made
                let was_corrected = output.corrected.is_some();
                let final_text = output.corrected.unwrap_or_else(|| result.text.clone());

                // Save to history only if correction was made
                if was_corrected {
                    if let Err(e) = config::Config::add_to_history(
                        &result.text,
                        &final_text,
                        &config.claude_model,
                        &config.custom_words,
                    ) {
                        eprintln!("Warning: Failed to save to history: {}", e);
                    }
                }

                // Display
                if args.debug {
                    if was_corrected {
                        eprintln!("Original:  {}", result.text);
                        eprintln!("Corrected: {}", final_text);
                        if let Some(explanation) = output.explanation {
                            eprintln!("Reason:    {}", explanation);
                        }
                        eprintln!();
                    } else {
                        eprintln!("No correction needed");
                        eprintln!();
                    }
                } else if was_corrected {
                    // Gray/dim for original, normal for corrected
                    eprintln!("\x1b[90m{}\x1b[0m", result.text);
                    eprintln!();
                }

                final_text
            }
            Err(e) => {
                eprintln!("\nClaude correction failed: {}", e);
                eprintln!("Falling back to original transcription\n");
                result.text
            }
        }
    } else {
        result.text
    };

    status("");
    println!("{}", final_text);

    if args.clip {
        Clipboard::new()?.set_text(&final_text)?;
    }

    Ok(())
}
