use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs;
use tracing::{info, warn};
use uuid::Uuid;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use crate::config;
use crate::openai::DreamSceneData;
use image;

// Support for commercial Stability API
#[derive(Debug, Serialize)]
struct StabilityApiRequest {
    text_prompts: Vec<TextPrompt>,
    cfg_scale: f32,
    height: u32,
    width: u32,
    samples: u32,
    steps: u32,
}

#[derive(Debug, Serialize)]
struct TextPrompt {
    text: String,
    weight: f32,
}

#[derive(Debug, Deserialize)]
struct StabilityApiResponse {
    artifacts: Vec<Artifact>,
}

#[derive(Debug, Deserialize)]
struct Artifact {
    base64: String,
    seed: u64,
    finish_reason: String,
}

// Support for Automatic1111 WebUI API (open source)
#[derive(Debug, Serialize)]
struct AutomaticApiRequest {
    prompt: String,
    negative_prompt: String,
    steps: u32,
    cfg_scale: f32,
    width: u32,
    height: u32,
    batch_size: u32,
    restore_faces: bool,
    sampler_name: String,
}

#[derive(Debug, Deserialize)]
struct AutomaticApiResponse {
    images: Vec<String>,
    parameters: serde_json::Value,
    info: String,
}

pub struct ImageGenerator {
    client: Client,
    output_dir: String,
}

impl ImageGenerator {
    pub fn new(output_dir: &str) -> Result<Self> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(output_dir)?;
        
        Ok(Self {
            client: Client::new(),
            output_dir: output_dir.to_string(),
        })
    }
    
    pub async fn generate_images(&self, scene_data: &DreamSceneData) -> Result<Vec<String>> {
        let cfg = config::get();
        
        // Create unique directory for this dream sequence
        let sequence_id = Uuid::new_v4().to_string();
        let sequence_dir = format!("{}/{}", self.output_dir, sequence_id);
        fs::create_dir_all(&sequence_dir)?;
        
        info!("Generating images for dream sequence: {}", scene_data.title);
        
        // Check if API key is valid
        if cfg.stable_diffusion.api_key.is_empty() || 
           cfg.stable_diffusion.api_key == "your_stability_api_key_here" ||
           cfg.stable_diffusion.api_key.starts_with("your_") {
            // Use mock images (solid color patterns) instead of real API
            info!("No valid Stable Diffusion API key, generating mock images");
            return self.generate_mock_images(scene_data, &sequence_dir);
        }
        
        // Construct prompt from scene data
        let mut prompt = format!(
            "{} - {}. {} mood with {} style. Features: ", 
            scene_data.title,
            scene_data.summary,
            scene_data.mood,
            scene_data.style_suggestion
        );
        
        // Add visual elements to prompt based on importance
        for element in &scene_data.elements {
            if element.importance > 5 {
                prompt.push_str(&format!("{} ({}), ", element.name, element.description));
            }
        }
        
        // Add color palette suggestion
        prompt.push_str(&format!("Color palette: {}", scene_data.color_palette.join(", ")));
        
        info!("Generated prompt: {}", prompt);
        
        // Use the appropriate API based on configuration
        let image_paths = if cfg.stable_diffusion.use_local_api {
            self.generate_with_automatic_api(prompt, &sequence_dir).await?
        } else {
            self.generate_with_stability_api(prompt, &sequence_dir).await?
        };
        
        Ok(image_paths)
    }
    
    async fn generate_with_stability_api(&self, prompt: String, sequence_dir: &str) -> Result<Vec<String>> {
        let cfg = config::get();
        
        let request = StabilityApiRequest {
            text_prompts: vec![
                TextPrompt {
                    text: prompt,
                    weight: 1.0,
                },
                TextPrompt {
                    text: "blurry, distorted, low quality, ugly, unrealistic".to_string(),
                    weight: -1.0,
                },
            ],
            cfg_scale: 7.0,
            height: 768,  // Reduced from 1024 to save memory
            width: 768,   // Reduced from 1024 to save memory
            samples: cfg.stable_diffusion.num_images,
            steps: cfg.stable_diffusion.steps,
        };
        
        let response = self.client
            .post(&cfg.stable_diffusion.api_endpoint)
            .header("Authorization", format!("Bearer {}", cfg.stable_diffusion.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await?
            .json::<StabilityApiResponse>()
            .await?;
        
        let mut image_paths = Vec::new();
        
        for (i, artifact) in response.artifacts.iter().enumerate() {
            if artifact.finish_reason != "SUCCESS" {
                warn!("Image generation issue with image {}: {}", i, artifact.finish_reason);
                continue;
            }
            
            let image_data = BASE64_STANDARD.decode(&artifact.base64)?;
            let image_path = format!("{}/image_{:03}.png", sequence_dir, i + 1);
            fs::write(&image_path, &image_data)?;
            
            info!("Saved image {} to {}", i + 1, image_path);
            image_paths.push(image_path);
        }
        
        Ok(image_paths)
    }
    
    async fn generate_with_automatic_api(&self, prompt: String, sequence_dir: &str) -> Result<Vec<String>> {
        let cfg = config::get();
        
        let request = AutomaticApiRequest {
            prompt,
            negative_prompt: "blurry, distorted, low quality, ugly, unrealistic".to_string(),
            steps: cfg.stable_diffusion.steps,
            cfg_scale: 7.0,
            width: 768,    // Reduced from 1024 to save memory
            height: 768,   // Reduced from 1024 to save memory
            batch_size: cfg.stable_diffusion.num_images,
            restore_faces: true,
            sampler_name: "Euler a".to_string(),
        };
        
        let response = self.client
            .post(&format!("{}/sdapi/v1/txt2img", cfg.stable_diffusion.local_api_endpoint))
            .json(&request)
            .send()
            .await?
            .json::<AutomaticApiResponse>()
            .await?;
        
        let mut image_paths = Vec::new();
        
        for (i, image_base64) in response.images.iter().enumerate() {
            let image_data = BASE64_STANDARD.decode(image_base64)?;
            let image_path = format!("{}/image_{:03}.png", sequence_dir, i + 1);
            fs::write(&image_path, &image_data)?;
            
            info!("Saved image {} to {}", i + 1, image_path);
            image_paths.push(image_path);
        }
        
        Ok(image_paths)
    }

    // Mock image generation when no API key is available
    fn generate_mock_images(&self, scene_data: &DreamSceneData, sequence_dir: &str) -> Result<Vec<String>> {
        let cfg = config::get();
        let mut image_paths = Vec::new();
        
        // Use color palette from scene data to create simple gradient images
        for i in 0..cfg.stable_diffusion.num_images {
            let image_path = format!("{}/image_{:03}.png", sequence_dir, i + 1);
            
            // Create a new image - 768x768 to match the normal size
            let width = 768;
            let height = 768;
            let mut img = image::RgbImage::new(width, height);
            
            // Get colors from palette (or use defaults if not enough)
            let colors = &scene_data.color_palette;
            let primary_color = Self::hex_to_rgb(colors.get(i as usize % colors.len()).unwrap_or(&"#87CEEB".to_string()));
            let secondary_color = Self::hex_to_rgb(colors.get((i as usize + 1) % colors.len()).unwrap_or(&"#FFFFFF".to_string()));
            
            // Fill image with a gradient based on colors
            for y in 0..height {
                for x in 0..width {
                    // Simple gradient calculation
                    let factor = (x as f32 / width as f32 + y as f32 / height as f32) / 2.0;
                    let r = (primary_color.0 as f32 * (1.0 - factor) + secondary_color.0 as f32 * factor) as u8;
                    let g = (primary_color.1 as f32 * (1.0 - factor) + secondary_color.1 as f32 * factor) as u8;
                    let b = (primary_color.2 as f32 * (1.0 - factor) + secondary_color.2 as f32 * factor) as u8;
                    
                    img.put_pixel(x, y, image::Rgb([r, g, b]));
                }
            }
            
            // Draw text with dream title
            // (This is a simplified version - real text rendering would require a font library)
            
            // Save the image
            img.save(&image_path)?;
            info!("Saved mock image {} to {}", i + 1, image_path);
            image_paths.push(image_path);
        }
        
        Ok(image_paths)
    }

    // Helper to convert hex color to RGB
    fn hex_to_rgb(hex: &str) -> (u8, u8, u8) {
        let hex = hex.trim_start_matches('#');
        
        match hex.len() {
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
                let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
                let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
                (r, g, b)
            },
            3 => {
                let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).unwrap_or(0);
                let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).unwrap_or(0);
                let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).unwrap_or(0);
                (r, g, b)
            },
            _ => (0, 0, 0) // Default to black
        }
    }
} 