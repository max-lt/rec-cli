//! Claude API correction for transcriptions

use crate::config::HistoryEntry;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ToolProperty {
    r#type: String,
    description: String,
}

#[derive(Serialize)]
struct ToolInputSchema {
    r#type: String,
    properties: std::collections::HashMap<String, ToolProperty>,
    required: Vec<String>,
}

#[derive(Serialize)]
struct Tool {
    name: String,
    description: String,
    input_schema: ToolInputSchema,
}

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    tool_choice: serde_json::Value,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum ContentBlock {
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct CorrectionResult {
    #[serde(default)]
    corrected: Option<String>,
    #[serde(default)]
    explanation: Option<String>,
}

pub struct CorrectionOutput {
    pub corrected: Option<String>,
    pub explanation: Option<String>,
}

/// Correct transcription using Claude API
pub async fn correct_transcription(
    text: &str,
    custom_words: &[String],
    model: &str,
    api_key: &str,
    history: &[HistoryEntry],
) -> Result<CorrectionOutput, Box<dyn std::error::Error>> {
    let custom_words_list = if custom_words.is_empty() {
        "(no custom words configured)".to_string()
    } else {
        custom_words.join(", ")
    };

    let context = if history.is_empty() {
        String::new()
    } else {
        let recent = history.iter().rev().take(5).rev();
        let mut ctx = String::from("\nContext (previous corrections):\n");
        for entry in recent {
            ctx.push_str(&format!(
                "- Original: \"{}\"\n  Corrected: \"{}\"\n",
                entry.original, entry.corrected
            ));
        }
        ctx.push('\n');
        ctx
    };

    let prompt = format!(
        r#"Correct this voice transcription taking into account the following technical terms:
{}
{}Rules:
- Only correct obvious mistakes and mistranscribed words
- Preserve the original punctuation and sentence structure
- Don't translate, don't rephrase, just correct
- If a word sounds like one from the list, use it
- Use the context from previous corrections to understand recurring style and terms

Original transcription:
{}

Use the 'report_correction' tool:
- If correction is needed: provide 'corrected' with the corrected text and 'explanation' with a brief reason
- If no correction is needed: call the tool with empty strings for both fields"#,
        custom_words_list, context, text
    );

    // Define the correction tool schema
    let mut properties = std::collections::HashMap::new();
    properties.insert(
        "corrected".to_string(),
        ToolProperty {
            r#type: "string".to_string(),
            description:
                "The corrected transcription text, or empty string if no correction needed"
                    .to_string(),
        },
    );
    properties.insert(
        "explanation".to_string(),
        ToolProperty {
            r#type: "string".to_string(),
            description: "Brief explanation of changes made, or empty string if no changes"
                .to_string(),
        },
    );

    let tool = Tool {
        name: "report_correction".to_string(),
        description: "Report the corrected transcription with optional explanation".to_string(),
        input_schema: ToolInputSchema {
            r#type: "object".to_string(),
            properties,
            required: vec!["corrected".to_string(), "explanation".to_string()],
        },
    };

    let request = ApiRequest {
        model: model.to_string(),
        max_tokens: 1024,
        messages: vec![Message {
            role: "user".to_string(),
            content: prompt,
        }],
        tools: vec![tool],
        tool_choice: serde_json::json!({"type": "tool", "name": "report_correction"}),
    };

    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !resp.status().is_success() {
        let body = resp.text().await?;
        return Err(format!("Claude API error: {}", body).into());
    }

    let body_text = resp.text().await?;

    let result: ApiResponse = serde_json::from_str(&body_text)
        .map_err(|e| format!("Failed to parse API response: {}\nBody: {}", e, body_text))?;

    // Find the tool_use content block
    let tool_input = result
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::ToolUse { input, .. } => Some(input),
            _ => None,
        })
        .ok_or("No tool_use in Claude response")?;

    // Parse the tool input as CorrectionResult
    let correction: CorrectionResult = serde_json::from_value(tool_input.clone())
        .map_err(|e| format!("Failed to parse tool input: {}", e))?;

    // If correction fields are empty, return None
    let corrected = correction.corrected.filter(|s| !s.is_empty());
    let explanation = correction.explanation.filter(|s| !s.is_empty());

    Ok(CorrectionOutput {
        corrected,
        explanation,
    })
}
