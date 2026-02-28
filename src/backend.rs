use reqwest::multipart;
use serde::Deserialize;

const MISTRAL_URL: &str = "https://api.mistral.ai/v1/audio/transcriptions";

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
}

pub struct TranscribeOptions {
    pub wav_data: Vec<u8>,
    pub model: String,
    pub language: Option<String>,
    pub context_bias: Vec<String>,
}

pub enum Backend {
    Mistral { api_key: String },
    RecApi { api_url: String, api_key: String },
}

impl Backend {
    pub async fn transcribe(
        &self,
        opts: TranscribeOptions,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match self {
            Backend::Mistral { api_key } => transcribe_mistral(&opts, api_key).await,
            Backend::RecApi { api_url, api_key } => {
                transcribe_rec_api(&opts, api_url, api_key).await
            }
        }
    }
}

async fn transcribe_mistral(
    opts: &TranscribeOptions,
    api_key: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(opts.wav_data.clone())
                .file_name("audio.wav")
                .mime_str("audio/wav")?,
        )
        .text("model", opts.model.clone());

    if let Some(lang) = &opts.language {
        form = form.text("language", lang.clone());
    }

    for term in &opts.context_bias {
        form = form.text("context_bias", term.clone());
    }

    let resp = client
        .post(MISTRAL_URL)
        .header("x-api-key", api_key)
        .multipart(form)
        .send()
        .await?;

    if !resp.status().is_success() {
        let body = resp.text().await?;
        return Err(format!("Mistral API error: {}", body).into());
    }

    let result: TranscriptionResponse = resp.json().await?;
    Ok(result.text)
}

async fn transcribe_rec_api(
    opts: &TranscribeOptions,
    api_url: &str,
    api_key: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/transcribe", api_url.trim_end_matches('/'));

    let mut form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(opts.wav_data.clone())
                .file_name("audio.wav")
                .mime_str("audio/wav")?,
        )
        .text("model", opts.model.clone());

    if let Some(lang) = &opts.language {
        form = form.text("language", lang.clone());
    }

    for term in &opts.context_bias {
        form = form.text("context_bias", term.clone());
    }

    let resp = client
        .post(&url)
        .header("authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await?;

    if !resp.status().is_success() {
        let body = resp.text().await?;
        return Err(format!("Rec API error: {}", body).into());
    }

    let result: TranscriptionResponse = resp.json().await?;
    Ok(result.text)
}
