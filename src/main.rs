//! rec - Quick speech-to-text for devs

mod backend;
mod config;
mod correction;
#[cfg(feature = "local")]
mod local;

use arboard::Clipboard;
use backend::Backend;
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use std::io::{self, BufWriter, IsTerminal, Write};
use std::sync::{Arc, Mutex};

/// Chunks of captured audio held for the model. Bounded, or a model that
/// falls behind grows the queue without limit.
#[cfg(feature = "local")]
const SAMPLE_QUEUE: usize = 128;

const MODEL_V1: &str = "voxtral-mini-2507";
const MODEL_V2: &str = "voxtral-mini-2602";

#[derive(Parser)]
#[command(name = "rec", version, about = "Quick speech-to-text for devs")]
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

    /// Language code (e.g. 'en', 'fr')
    #[arg(short = 'l', long, alias = "lang", global = true)]
    language: Option<String>,

    /// Send custom words as context_bias to Mistral
    #[arg(short = 'b', long, global = true)]
    bias: bool,

    /// Transcribe with the local MLX Voxtral model, no API call
    #[cfg(feature = "local")]
    #[arg(long, global = true, conflicts_with_all = ["v2", "language"])]
    local: bool,

    /// Read 16 kHz mono 16-bit PCM from stdin, and encode while it arrives
    #[cfg(feature = "local")]
    #[arg(long, global = true, requires = "local", conflicts_with = "file")]
    pcm: bool,

    /// Print text continuously, for as long as audio arrives
    #[cfg(feature = "local")]
    #[arg(long, global = true, requires = "local", conflicts_with = "correct")]
    captions: bool,

    /// Record the whole take before transcribing it, instead of encoding
    /// while the microphone runs
    #[cfg(feature = "local")]
    #[arg(long, global = true, requires = "local", conflicts_with = "pcm")]
    batch: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a custom word to the vocabulary (for Claude correction)
    AddWord { word: String },
}

#[cfg_attr(not(feature = "local"), allow(unused_variables))]
fn select_backend(args: &Args) -> Result<Backend, Box<dyn std::error::Error>> {
    #[cfg(feature = "local")]
    if args.local {
        let weights =
            std::env::var("REC_LOCAL_WEIGHTS").unwrap_or_else(|_| local::DEFAULT_WEIGHTS.into());
        return Ok(Backend::Local { weights });
    }

    let rec_api_key = std::env::var("REC_API_KEY").ok();
    let rec_api_url = std::env::var("REC_API_URL").ok();
    let mistral_key = std::env::var("MISTRAL_API_KEY").ok();

    if let (Some(api_key), Some(api_url)) = (rec_api_key, rec_api_url) {
        Ok(Backend::RecApi { api_url, api_key })
    } else if let Some(api_key) = mistral_key {
        Ok(Backend::Mistral { api_key })
    } else {
        Err("Set REC_API_KEY + REC_API_URL or MISTRAL_API_KEY".into())
    }
}

/// Custom words, split into single tokens for context bias.
fn bias_words(args: &Args, config: &config::Config) -> Vec<String> {
    if !args.bias {
        return vec![];
    }
    config
        .custom_words
        .iter()
        .flat_map(|w| {
            w.split(':')
                .next()
                .unwrap_or(w)
                .split_whitespace()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        })
        .filter(|w| !w.is_empty())
        .collect()
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

    // Continuous captions come first: they take a source that the other
    // branches would claim.
    #[cfg(feature = "local")]
    if args.captions {
        return run_captions(&args).await;
    }

    // The caller records and pushes the samples in, so the encoder runs
    // during the capture and only the decoder is left at the end.
    #[cfg(feature = "local")]
    if args.pcm {
        let config = config::Config::load()?;
        let weights =
            std::env::var("REC_LOCAL_WEIGHTS").unwrap_or_else(|_| local::DEFAULT_WEIGHTS.into());
        if io::stdin().is_terminal() {
            return Err("--pcm reads audio from stdin: pipe 16 kHz mono 16-bit PCM into it".into());
        }
        status("Reading audio...");
        let text =
            local::transcribe_stream(io::stdin().lock(), &weights, &bias_words(&args, &config))?;
        return finish(text, &args, &config).await;
    }

    // A file, through the same encoder a capture uses. Nothing overlaps, so
    // it saves no time; it exists to compare both paths on equal audio.
    #[cfg(feature = "local")]
    if args.local && !args.batch && let Some(path) = args.file.clone() {
        let config = config::Config::load()?;
        let weights =
            std::env::var("REC_LOCAL_WEIGHTS").unwrap_or_else(|_| local::DEFAULT_WEIGHTS.into());
        status("Reading file...");
        let bytes = std::fs::read(path)?;
        status("Transcribing...");
        let text = local::transcribe_chunked(&bytes, &weights, &bias_words(&args, &config))?;
        return finish(text, &args, &config).await;
    }

    // Local microphone capture, encoded while it runs. --batch keeps the old
    // path, which records the whole take first.
    #[cfg(feature = "local")]
    if args.local && !args.batch && args.file.is_none() {
        let config = config::Config::load()?;
        let weights =
            std::env::var("REC_LOCAL_WEIGHTS").unwrap_or_else(|_| local::DEFAULT_WEIGHTS.into());
        let bias = bias_words(&args, &config);

        status("Loading...");
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or("No mic")?;
        let device_config = device.default_input_config()?;
        let rate = device_config.sample_rate();
        let channels = device_config.channels() as usize;

        let (sender, receiver) = std::sync::mpsc::sync_channel::<Vec<f32>>(SAMPLE_QUEUE);
        // MLX stays on one thread, and this one owns the model.
        let worker =
            std::thread::spawn(move || local::transcribe_live(receiver, rate, &weights, &bias));

        let stream = match device_config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &device_config.into(),
                move |data: &[f32], _: &_| {
                    offer(&sender, downmix(data, channels));
                },
                |err| eprintln!("Error: {}", err),
                None,
            )?,
            cpal::SampleFormat::I16 => device.build_input_stream(
                &device_config.into(),
                move |data: &[i16], _: &_| {
                    let floats: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    offer(&sender, downmix(&floats, channels));
                },
                |err| eprintln!("Error: {}", err),
                None,
            )?,
            _ => return Err("Unsupported format".into()),
        };

        stream.play()?;
        status("Recording...");

        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        // Dropping the stream drops the sender, which ends the worker loop.
        drop(stream);

        status_up("Transcribing...");
        let text = worker
            .join()
            .map_err(|_| "recorder thread panicked")?
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        return finish(text, &args, &config).await;
    }

    let backend = select_backend(&args)?;

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

    let config = config::Config::load()?;

    let model = if args.v2 { MODEL_V2 } else { MODEL_V1 };
    let text = backend
        .transcribe(backend::TranscribeOptions {
            wav_data: wav_buffer,
            model: model.to_string(),
            language: args.language.clone(),
            context_bias: bias_words(&args, &config),
        })
        .await?;

    finish(text, &args, &config).await
}

/// Correction, clipboard and stdout, shared by every input path.
async fn finish(
    text: String,
    args: &Args,
    config: &config::Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let final_text = if args.correct {
        status("Correcting with Claude...");

        let anthropic_key =
            std::env::var("ANTHROPIC_API_KEY").map_err(|_| "ANTHROPIC_API_KEY not set")?;

        let history = config::Config::load_history().unwrap_or_default();

        match correction::correct_transcription(
            &text,
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
                let final_text = output.corrected.unwrap_or_else(|| text.clone());

                // Save to history only if correction was made
                if was_corrected {
                    if let Err(e) = config::Config::add_to_history(
                        &text,
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
                        eprintln!("Original:  {}", text);
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
                    eprintln!("\x1b[90m{}\x1b[0m", text);
                    eprintln!();
                }

                final_text
            }
            Err(e) => {
                eprintln!("\nClaude correction failed: {}", e);
                eprintln!("Falling back to original transcription\n");
                text
            }
        }
    } else {
        text
    };

    status("");
    println!("{}", final_text);

    if args.clip {
        Clipboard::new()?.set_text(&final_text)?;
    }

    Ok(())
}

/// Averages the channels of an interleaved buffer into mono.
#[cfg(feature = "local")]
fn downmix(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}
/// `--captions`: prints text for as long as audio arrives, from the
/// microphone, from stdin or from a file.
#[cfg(feature = "local")]
async fn run_captions(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let weights =
        std::env::var("REC_LOCAL_WEIGHTS").unwrap_or_else(|_| local::DEFAULT_WEIGHTS.into());
    let (sender, receiver) = std::sync::mpsc::sync_channel::<Vec<f32>>(SAMPLE_QUEUE);

    if let Some(path) = args.file.clone() {
        let bytes = std::fs::read(path)?;
        std::thread::spawn(move || local::send_file(&bytes, &sender).ok());
        return print_captions(receiver, 16_000, &weights, args);
    }

    if args.pcm {
        if io::stdin().is_terminal() {
            return Err("--pcm reads audio from stdin: pipe 16 kHz mono 16-bit PCM into it".into());
        }
        std::thread::spawn(move || local::send_pcm(io::stdin().lock(), &sender).ok());
        return print_captions(receiver, 16_000, &weights, args);
    }

    status("Loading...");
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("No mic")?;
    let device_config = device.default_input_config()?;
    let rate = device_config.sample_rate();
    let channels = device_config.channels() as usize;

    // The captions run on their own thread, so Enter can drop the stream here
    // and let the decoder flush what it still holds.
    let owned = weights.clone();
    let worker = std::thread::spawn(move || local::caption(receiver, rate, &owned));

    let stream = match device_config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &device_config.into(),
            move |data: &[f32], _: &_| {
                offer(&sender, downmix(data, channels));
            },
            |err| eprintln!("Error: {}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &device_config.into(),
            move |data: &[i16], _: &_| {
                let floats: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                offer(&sender, downmix(&floats, channels));
            },
            |err| eprintln!("Error: {}", err),
            None,
        )?,
        _ => return Err("Unsupported format".into()),
    };

    stream.play()?;
    status("");

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    drop(stream);

    let text = worker
        .join()
        .map_err(|_| "captions thread panicked")?
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    if args.clip && !text.is_empty() {
        Clipboard::new()?.set_text(&text)?;
    }
    Ok(())
}

#[cfg(feature = "local")]
fn print_captions(
    receiver: std::sync::mpsc::Receiver<Vec<f32>>,
    rate: u32,
    weights: &str,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let text = local::caption(receiver, rate, weights)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    if args.clip && !text.is_empty() {
        Clipboard::new()?.set_text(&text)?;
    }
    Ok(())
}
/// Hands captured audio to the model, and drops it when the model is behind.
/// Blocking here would stall the audio callback and glitch the capture.
#[cfg(feature = "local")]
fn offer(sender: &std::sync::mpsc::SyncSender<Vec<f32>>, samples: Vec<f32>) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);

    if sender.try_send(samples).is_err() && !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("\nrec: the model is behind, audio is being dropped");
    }
}
