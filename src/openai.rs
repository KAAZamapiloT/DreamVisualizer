use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;
use crate::config;

// Common structures for both OpenAI and local LLM APIs
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: MessageResponse,
}

#[derive(Debug, Deserialize)]
struct MessageResponse {
    content: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DreamSceneData {
    pub title: String,
    pub summary: String,
    pub elements: Vec<VisualElement>,
    pub mood: String,
    pub color_palette: Vec<String>,
    pub style_suggestion: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VisualElement {
    pub name: String,
    pub description: String,
    pub importance: u8,
}

pub async fn process_dream_description(dream_text: &str) -> Result<DreamSceneData> {
    let cfg = config::get();
    let client = Client::new();
    
    // Check if API key is a placeholder or empty
    if cfg.openai.api_key.is_empty() ||
       cfg.openai.api_key == "your_openai_api_key_here" ||
       cfg.openai.api_key.starts_with("your_") {
        // Use mock data instead of API call
        info!("No valid API key found, using mock dream interpretation");
        return Ok(generate_mock_dream_data(dream_text));
    }
    
    let system_message = "You are a visual interpreter that extracts and enhances visual elements from dreams. \
        Analyze the dream description and return a structured JSON with these elements: \
        1. title (short, evocative title for the dream) \
        2. summary (1-2 sentence summary) \
        3. elements (array of objects with name, description, and importance 1-10) \
        4. mood (overall emotional tone) \
        5. color_palette (array of 3-5 hex colors that match the mood) \
        6. style_suggestion (artistic style that would fit this scene)";
    
    let user_message = format!("Here is my dream to interpret: {}", dream_text);
    
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: system_message.to_string(),
        },
        Message {
            role: "user".to_string(),
            content: user_message,
        },
    ];

    // Force using OpenAI API directly since local LLM is not running
    let content = process_with_openai(&client, messages).await?;
    
    // Extract JSON from the response content
    let content = if content.contains("```json") {
        let start = content.find("```json").unwrap() + 7;
        let end = content.rfind("```").unwrap();
        &content[start..end]
    } else if content.contains("```") {
        let start = content.find("```").unwrap() + 3;
        let end = content.rfind("```").unwrap();
        &content[start..end]
    } else {
        &content
    };
    
    let scene_data: DreamSceneData = serde_json::from_str(content)?;
    Ok(scene_data)
}

async fn process_with_openai(client: &Client, messages: Vec<Message>) -> Result<String> {
    let cfg = config::get();

    let request = ChatCompletionRequest {
        model: cfg.openai.model.clone(),
        messages,
        max_tokens: Some(cfg.openai.max_tokens),
        temperature: Some(0.7),
    };
    
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", cfg.openai.api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;
    
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        anyhow::bail!("OpenAI API error: {} - {}", status, error_text);
    }
    
    // Parse the response body
    let body = response.text().await?;
    
    // Try to parse as ChatCompletionResponse
    match serde_json::from_str::<ChatCompletionResponse>(&body) {
        Ok(parsed) => {
            if parsed.choices.is_empty() {
                anyhow::bail!("No response choices returned from OpenAI API");
            }
            Ok(parsed.choices[0].message.content.clone())
        },
        Err(err) => {
            // Log the error and the response body for debugging
            anyhow::bail!("Failed to parse OpenAI response: {} - Response: {}", err, body)
        }
    }
}

// Mock function to generate dream data without API
fn generate_mock_dream_data(dream_text: &str) -> DreamSceneData {
    // Extract a simple title from the dream text (first few words)
    let title = dream_text
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ") + "...";
    
    // Create a default dream scene data
    DreamSceneData {
        title,
        summary: format!("A dream sequence featuring {}.", dream_text),
        elements: vec![
            VisualElement {
                name: "Clouds".to_string(),
                description: "Fluffy white clouds stretching across a blue sky".to_string(),
                importance: 9,
            },
            VisualElement {
                name: "Landscape".to_string(),
                description: "Vast, beautiful landscape with rolling hills".to_string(),
                importance: 8,
            },
            VisualElement {
                name: "Mountains".to_string(),
                description: "Majestic peaks rising in the distance".to_string(),
                importance: 7,
            },
            VisualElement {
                name: "Rivers".to_string(),
                description: "Winding rivers cutting through the terrain".to_string(),
                importance: 6,
            },
        ],
        mood: "Peaceful and expansive".to_string(),
        color_palette: vec![
            "#87CEEB".to_string(), // Sky blue
            "#FFFFFF".to_string(), // White
            "#228B22".to_string(), // Forest green
            "#8B4513".to_string(), // Saddle brown
            "#4682B4".to_string(), // Steel blue
        ],
        style_suggestion: "Panoramic landscape painting".to_string(),
    }
} 