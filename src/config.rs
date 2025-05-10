use anyhow::Result;
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub openai: OpenAiConfig,
    pub stable_diffusion: StableDiffusionConfig,
    pub server: ServerConfig,
    pub ffmpeg: FfmpegConfig,
    pub local_llm: LocalLLMConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StableDiffusionConfig {
    pub api_endpoint: String,
    pub api_key: String,
    pub num_images: u32,
    pub steps: u32,
    pub use_local_api: bool,
    pub local_api_endpoint: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LocalLLMConfig {
    pub enabled: bool,
    pub api_endpoint: String,
    pub model: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FfmpegConfig {
    pub fps: u32,
    pub duration: u32,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

pub fn get() -> &'static Config {
    CONFIG.get().expect("Config not initialized")
}

pub fn load() -> Result<()> {
    dotenv::dotenv().ok();
    
    let config = Config {
        openai: OpenAiConfig {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string()),
            max_tokens: std::env::var("OPENAI_MAX_TOKENS")
                .unwrap_or_else(|_| "1024".to_string())
                .parse()
                .unwrap_or(1024),
        },
        stable_diffusion: StableDiffusionConfig {
            api_endpoint: std::env::var("SD_API_ENDPOINT")
                .unwrap_or_else(|_| "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image".to_string()),
            api_key: std::env::var("SD_API_KEY").unwrap_or_default(),
            num_images: std::env::var("SD_NUM_IMAGES")
                .unwrap_or_else(|_| "4".to_string())
                .parse()
                .unwrap_or(4),
            steps: std::env::var("SD_STEPS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            use_local_api: std::env::var("SD_USE_LOCAL_API")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
            local_api_endpoint: std::env::var("SD_LOCAL_API_ENDPOINT")
                .unwrap_or_else(|_| "http://127.0.0.1:7860".to_string()),
        },
        local_llm: LocalLLMConfig {
            enabled: std::env::var("LOCAL_LLM_ENABLED")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
            api_endpoint: std::env::var("LOCAL_LLM_API_ENDPOINT")
                .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string()),
            model: std::env::var("LOCAL_LLM_MODEL")
                .unwrap_or_else(|_| "llama3".to_string()),
        },
        server: ServerConfig {
            host: std::env::var("SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("SERVER_PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .unwrap_or(3000),
        },
        ffmpeg: FfmpegConfig {
            fps: std::env::var("FFMPEG_FPS")
                .unwrap_or_else(|_| "5".to_string())
                .parse()
                .unwrap_or(5),
            duration: std::env::var("FFMPEG_DURATION")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .unwrap_or(3),
        },
    };

    CONFIG.set(config).expect("Failed to set config");
    Ok(())
} 