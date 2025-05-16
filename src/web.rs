use anyhow::Result;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{Html, IntoResponse, Redirect, Response},
    routing::{get, post},
    Router,
    Json,
};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tera::{Context, Tera};
use tokio::fs;
use tower_http::services::ServeDir;
use tracing::{error, info};
use uuid::Uuid;
use std::process::Command;
use base64;

use crate::config;
use crate::image::ImageGenerator;
use crate::openai::{self, DreamSceneData};
use crate::video::VideoGenerator;
use crate::learning::{self, LearningSystem, FeedbackData, LearningStatistics};

// Global learning system (would be better handled with proper dependency injection)
static mut LEARNING_SYSTEM: Option<Arc<LearningSystem>> = None;

// Initialize the learning system
pub fn init_learning_system(system: LearningSystem) {
    unsafe {
        LEARNING_SYSTEM = Some(Arc::new(system));
    }
}

// Get reference to learning system
fn get_learning_system() -> Option<Arc<LearningSystem>> {
    unsafe {
        LEARNING_SYSTEM.as_ref().map(|sys| sys.clone())
    }
}

// Struct to represent temporarily buffered content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferedContent {
    id: String,
    content_type: String, // "image", "gif", "story"
    data: String, // Base64 encoded data or text content
    metadata: serde_json::Value, // Additional metadata
    timestamp: chrono::DateTime<chrono::Utc>,
    // Expires after a certain time
    expires_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone)]
pub struct AppState {
    templates: Tera,
    dream_data: Arc<Mutex<HashMap<String, DreamData>>>,
    // Add a buffer for temporary content
    temp_buffer: Arc<Mutex<HashMap<String, BufferedContent>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DreamData {
    id: String,
    input_text: String,
    processed_data: DreamSceneData,
    video_path: Option<String>,
    image_paths: Vec<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
    model_used: Option<String>,
    feedback: Option<FeedbackInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackInfo {
    helpful: bool,
    comments: Option<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Deserialize)]
struct StoryGenRequest {
    prompt: String,
    story_type: String,
}

#[derive(Deserialize)]
struct ImageGenRequest {
    prompt: String,
    quality: Option<String>,
    model: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    guidance_scale: Option<f32>,
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct GifGenRequest {
    prompt: String,
    num_frames: Option<u32>,
    fps: Option<f32>,
    seed: Option<u64>,
    model: Option<String>,
    format: Option<String>,       // "gif", "mp4", or "both"
    guidance_scale: Option<f32>,  // Controls how closely the model follows the prompt
    width: Option<u32>,           // Image width
    height: Option<u32>,          // Image height
    quality: Option<String>,      // "standard", "high", or "ultra"
    interpolation_steps: Option<u32>,  // Number of interpolation steps between keyframes
    animation_effect: Option<String>,  // "none", "zoom-in", "zoom-out", "pan-left", "pan-right", "dissolve", "morph", "random"
    transition_type: Option<String>,   // "smooth", "fade", "wipe", "dissolve"
    narration: Option<NarrationRequest>, // Settings for generating narration
}

#[derive(Deserialize)]
struct NarrationRequest {
    enable: bool,
    narration_type: Option<String>,  // "story", "interpretation", "poetic"
    style: Option<String>,           // "neutral", "dramatic", "whimsical", "mysterious"
    subtitles: Option<bool>,
    background_music: Option<bool>,
}

#[derive(Deserialize)]
struct FeedbackRequest {
    dream_id: String,
    helpful: bool,
    comments: Option<String>,
}

#[derive(Deserialize)]
struct BufferContentRequest {
    content_type: String,
    data: String,
    metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct SaveBufferedContentRequest {
    buffer_id: String,
    filename: Option<String>,
}

#[derive(Deserialize)]
struct FineTuneRequest {
    base_model: String,
    output_model: String,
}

#[derive(Deserialize)]
struct EnhancedImageGenRequest {
    prompt: String,
    models: Option<String>,         // Comma-separated model names
    blend_mode: Option<String>,     // Blending mode
    quality: Option<String>,
    guidance_scale: Option<f32>,
    width: Option<u32>,
    height: Option<u32>,
    seed: Option<u64>,
    enhancement: Option<f32>,       // Enhancement level
    save_all: Option<bool>,         // Whether to save all intermediate images
}

#[derive(Deserialize)]
struct ModifyImageRequest {
    prompt: String,
    image_id: String,       // ID of the uploaded temporary image
    model: Option<String>,  // Model to use
    strength: Option<f32>,  // Strength of modification
    steps: Option<u32>,     // Number of inference steps
    guidance_scale: Option<f32>,  // Guidance scale
    seed: Option<u64>,      // Random seed
    enhancement: Option<f32>, // Enhancement level
}

#[derive(Deserialize)]
struct InpaintImageRequest {
    prompt: String,
    image_id: String,       // ID of the uploaded image
    mask_id: String,        // ID of the uploaded mask
    model: Option<String>,  // Model to use
    steps: Option<u32>,     // Number of inference steps
    guidance_scale: Option<f32>,  // Guidance scale
    seed: Option<u64>,      // Random seed
    enhancement: Option<f32>, // Enhancement level
}

// Add this import for the middleware
use tower::ServiceBuilder;

pub async fn start_server() -> Result<()> {
    let cfg = config::get();
    
    // Set up Tera templates
    let mut tera = Tera::default();
    tera.add_raw_templates(vec![
        ("index.html", include_str!("../templates/index.html")),
        ("result.html", include_str!("../templates/result.html")),
        ("history.html", include_str!("../templates/history.html")),
        ("learning.html", include_str!("../templates/learning.html")),
        ("enhanced_images.html", include_str!("../templates/enhanced_images.html")),
        ("modify_image.html", include_str!("../templates/modify_image.html")),
        ("inpaint_image.html", include_str!("../templates/inpaint_image.html")),
    ])?;
    
    // Create directories for storing generated content
    let data_dir = Path::new("data");
    let images_dir = data_dir.join("images");
    let videos_dir = data_dir.join("videos");
    let stories_dir = data_dir.join("stories");
    let models_dir = data_dir.join("models");
    let uploads_dir = data_dir.join("uploads");
    let modified_dir = data_dir.join("modified");
    let inpainted_dir = data_dir.join("inpainted");
    
    fs::create_dir_all(&images_dir).await?;
    fs::create_dir_all(&videos_dir).await?;
    fs::create_dir_all(&stories_dir).await?;
    fs::create_dir_all(&models_dir).await?;
    fs::create_dir_all(&uploads_dir).await?;
    fs::create_dir_all(&modified_dir).await?;
    fs::create_dir_all(&inpainted_dir).await?;
    
    // Set up app state
    let app_state = AppState {
        templates: tera,
        dream_data: Arc::new(Mutex::new(HashMap::new())),
        temp_buffer: Arc::new(Mutex::new(HashMap::new())),
    };
    
    // Start a background task to clean up expired content from temp_buffer
    let _buffer_cleaner = {
        let temp_buffer = app_state.temp_buffer.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600)); // Run every hour
            loop {
                interval.tick().await;
                cleanup_expired_content(temp_buffer.clone());
            }
        })
    };
    
    // Build our application with routes
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/process", post(process_handler))
        .route("/process_local", post(process_local_handler))
        .route("/generate_story", post(generate_story_handler))
        .route("/generate_images_local", post(generate_images_local_handler))
        .route("/generate_gif", post(generate_gif_handler))
        .route("/submit_feedback", post(submit_feedback_handler))
        .route("/buffer_content", post(buffer_content_handler))
        .route("/save_buffered_content", post(save_buffered_content_handler))
        .route("/get_buffered_content/:id", get(get_buffered_content_handler))
        .route("/learning", get(learning_handler))
        .route("/learning/stats", get(learning_stats_handler))
        .route("/learning/fine_tune", post(fine_tune_handler))
        .route("/result/:id", get(result_handler))
        .route("/history", get(history_handler))
        .route("/video", get(video_generation_handler))
        .route("/images", get(image_generation_handler))
        .route("/generate_enhanced_images", post(generate_enhanced_images_handler))
        .route("/enhanced", get(enhanced_image_generation_handler))
        .route("/upload_image_for_modification", post(upload_image_handler))
        .route("/modify_image", post(simple_modify_image_handler))
        .route("/modify", get(modify_image_page_handler))
        .route("/upload_mask", post(upload_mask_handler))
        .route("/inpaint_image", post(simple_inpaint_image_handler))
        .route("/inpaint", get(inpaint_image_page_handler))
        .nest_service("/static", ServeDir::new("static"))
        .nest_service("/data", ServeDir::new("data"))
        .with_state(app_state);
    
    // Start the server
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    info!("Starting server at http://{}", addr);
    
    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}

async fn index_handler(State(state): State<AppState>) -> impl IntoResponse {
    let context = Context::new();
    match state.templates.render("index.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(err) => {
            error!("Template error: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn process_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Extract the dream description from the form
    let mut dream_text = String::new();
    
    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        if field.name().unwrap_or("") == "dream_description" {
            dream_text = field.text().await.unwrap_or_default();
            break;
        }
    }
    
    if dream_text.is_empty() {
        return (StatusCode::BAD_REQUEST, "Dream description is required").into_response();
    }
    
    // Generate a unique ID for this dream visualization
    let dream_id = Uuid::new_v4().to_string();
    
    // Process the dream description with OpenAI to extract visual elements
    let processed_data = match openai::process_dream_description(&dream_text).await {
        Ok(data) => data,
        Err(err) => {
            error!("Failed to process dream description: {}", err);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to process dream description").into_response();
        }
    };
    
    // Generate images based on the processed data
    let image_generator = match ImageGenerator::new("data/images") {
        Ok(gen) => gen,
        Err(err) => {
            error!("Failed to create image generator: {}", err);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to create image generator").into_response();
        }
    };
    
    let image_paths = match image_generator.generate_images(&processed_data).await {
        Ok(paths) => paths,
        Err(err) => {
            error!("Failed to generate images: {}", err);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to generate images").into_response();
        }
    };
    
    // Generate video from the images
    let video_generator = VideoGenerator::new();
    
    // Try WebP first (higher quality), fall back to GIF if it fails
    let video_path = match video_generator.generate_webp_animation(&image_paths, "data/videos") {
        Ok(path) => Some(path),
        Err(err) => {
            error!("Failed to generate WebP animation: {}", err);
            // Fall back to GIF
            match video_generator.generate_video(&image_paths, "data/videos") {
                Ok(path) => Some(path),
                Err(err) => {
                    error!("Failed to generate GIF animation: {}", err);
                    None
                }
            }
        }
    };
    
    // Store the dream data
    let dream_data = DreamData {
        id: dream_id.clone(),
        input_text: dream_text,
        processed_data,
        video_path,
        image_paths,
        timestamp: chrono::Utc::now(),
        model_used: None,
        feedback: None,
    };
    
    state.dream_data.lock().unwrap().insert(dream_id.clone(), dream_data);
    
    // Redirect to the result page
    Redirect::to(&format!("/result/{}", dream_id)).into_response()
}

async fn process_local_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Extract the dream description from the form
    let mut dream_text = String::new();
    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        if field.name().unwrap_or("") == "dream_description" {
            dream_text = field.text().await.unwrap_or_default();
            break;
        }
    }
    if dream_text.is_empty() {
        return (StatusCode::BAD_REQUEST, "Dream description is required").into_response();
    }
    // Generate a unique ID for this dream visualization
    let dream_id = Uuid::new_v4().to_string();
    let output_dir = format!("data/images/{}", dream_id);
    // Call the Python script to generate images
    let python_status = Command::new("python3")
        .arg("generate_dream_images.py")
        .arg("--prompt").arg(&dream_text)
        .arg("--output_dir").arg(&output_dir)
        .arg("--num_images").arg("4")
        .status();
    match python_status {
        Ok(status) if status.success() => {},
        Ok(status) => {
            error!("Python script failed with status: {}", status);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Image generation failed").into_response();
        },
        Err(e) => {
            error!("Failed to start Python script: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Image generation failed").into_response();
        }
    }
    // Collect generated image paths
    let mut image_paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&output_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "png" {
                    image_paths.push(path.display().to_string());
                }
            }
        }
        image_paths.sort();
    }
    // Use mock dream interpretation for now (or you can call OpenAI/local LLM as needed)
    let processed_data = openai::process_dream_description(&dream_text).await.unwrap_or_else(|_| openai::DreamSceneData {
        title: "Dream".to_string(),
        summary: dream_text.clone(),
        elements: vec![],
        mood: "Unknown".to_string(),
        color_palette: vec!["#cccccc".to_string()],
        style_suggestion: "None".to_string(),
    });
    let dream_data = DreamData {
        id: dream_id.clone(),
        input_text: dream_text,
        processed_data,
        video_path: None,
        image_paths: image_paths.clone(),
        timestamp: chrono::Utc::now(),
        model_used: None,
        feedback: None,
    };
    state.dream_data.lock().unwrap().insert(dream_id.clone(), dream_data);
    Redirect::to(&format!("/result/{}", dream_id)).into_response()
}

async fn result_handler(
    State(state): State<AppState>,
    axum::extract::Path(dream_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let dream_data_lock = state.dream_data.lock().unwrap();
    let dream_data = match dream_data_lock.get(&dream_id) {
        Some(data) => data,
        None => return StatusCode::NOT_FOUND.into_response(),
    };
    
    let mut context = Context::new();
    context.insert("dream", dream_data);
    
    // Prepare relative paths for browser
    if let Some(video_path) = &dream_data.video_path {
        let relative_path = video_path.replace("data/", "/data/");
        context.insert("video_path", &relative_path);
    }
    
    let relative_image_paths: Vec<String> = dream_data
        .image_paths
        .iter()
        .map(|path| path.replace("data/", "/data/"))
        .collect();
    context.insert("image_paths", &relative_image_paths);
    context.insert("images", &relative_image_paths);
    
    match state.templates.render("result.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(err) => {
            error!("Template error: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn history_handler(State(state): State<AppState>) -> impl IntoResponse {
    let dream_data_lock = state.dream_data.lock().unwrap();
    
    let mut context = Context::new();
    let dreams: Vec<&DreamData> = dream_data_lock.values().collect();
    context.insert("dreams", &dreams);
    
    match state.templates.render("history.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(err) => {
            error!("Template error: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn generate_story_handler(
    Json(payload): Json<StoryGenRequest>,
) -> impl IntoResponse {
    // Validate story_type
    if !["sequel", "prequel", "continuation"].contains(&payload.story_type.as_str()) {
        return (StatusCode::BAD_REQUEST, "Invalid story type").into_response();
    }

    // Call the Python script to generate a story
    let output = Command::new("python3")
        .arg("generate_dream_story.py")
        .arg("--prompt").arg(&payload.prompt)
        .arg("--type").arg(&payload.story_type)
        .output();

    match output {
        Ok(output) if output.status.success() => {
            // Parse the JSON output from the script
            match String::from_utf8(output.stdout) {
                Ok(stdout) => {
                    match serde_json::from_str::<serde_json::Value>(&stdout) {
                        Ok(json) => Json(json).into_response(),
                        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Failed to parse story output").into_response(),
                    }
                },
                Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Invalid output from story generator").into_response(),
            }
        },
        Ok(output) => {
            // Script ran but returned an error
            error!("Story generation script failed: {}", String::from_utf8_lossy(&output.stderr));
            (StatusCode::INTERNAL_SERVER_ERROR, "Story generation failed").into_response()
        },
        Err(e) => {
            // Failed to run script
            error!("Failed to run story generation script: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to generate story").into_response()
        }
    }
}

async fn generate_images_local_handler(
    Json(payload): Json<ImageGenRequest>,
) -> impl IntoResponse {
    let dream_id = Uuid::new_v4().to_string();
    let output_dir = format!("data/images/{}_gpu", dream_id);
    
    // Set up command with all parameters
    let mut command = Command::new("python3");
    command.arg("generate_dream_images.py")
        .arg("--prompt").arg(&payload.prompt)
        .arg("--output_dir").arg(&output_dir)
        .arg("--num_images").arg("4");
    
    // Add optional parameters if provided
    if let Some(model) = &payload.model {
        command.arg("--model").arg(model);
    }
    
    if let Some(quality) = &payload.quality {
        command.arg("--quality").arg(quality);
    }
    
    if let Some(width) = payload.width {
        command.arg("--width").arg(&width.to_string());
    }
    
    if let Some(height) = payload.height {
        command.arg("--height").arg(&height.to_string());
    }
    
    if let Some(guidance_scale) = payload.guidance_scale {
        command.arg("--guidance_scale").arg(&guidance_scale.to_string());
    }
    
    if let Some(seed) = payload.seed {
        command.arg("--seed").arg(&seed.to_string());
    }
    
    // Run the command
    let python_status = command.status();
    
    match python_status {
        Ok(status) if status.success() => {},
        Ok(status) => {
            error!("Python script failed with status: {}", status);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Image generation failed").into_response();
        },
        Err(e) => {
            error!("Failed to start Python script: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Image generation failed").into_response();
        }
    }
    
    // Collect generated image paths
    let mut image_paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&output_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "png" {
                    let rel_path = path.display().to_string().replace("data/", "/data/");
                    image_paths.push(rel_path);
                }
            }
        }
        image_paths.sort();
    }
    
    axum::Json(serde_json::json!({
        "images": image_paths,
        "quality": payload.quality.unwrap_or_else(|| "high".to_string())
    })).into_response()
}

async fn generate_gif_handler(
    State(_state): State<AppState>,
    Json(payload): Json<GifGenRequest>,
) -> impl IntoResponse {
    let dream_id = Uuid::new_v4().to_string();
    let output_dir = format!("data/videos/{}_animation", dream_id);
    let base_filename = format!("{}/animation", output_dir);
    
    // Ensure output directory exists
    if let Err(e) = tokio::fs::create_dir_all(&output_dir).await {
        error!("Failed to create output directory: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to create output directory").into_response();
    }
    
    // Set default values for optional parameters
    let num_frames = payload.num_frames.unwrap_or(4);
    let fps = payload.fps.unwrap_or(10.0);
    let format = payload.format.as_deref().unwrap_or("gif");
    let quality = payload.quality.as_deref().unwrap_or("high");
    
    // Build command with additional parameters for improved quality
    let mut command = Command::new("python3");
    command.arg("generate_dream_gif.py")
        .arg("--prompt").arg(&payload.prompt)
        .arg("--output_path").arg(&base_filename)
        .arg("--num_frames").arg(&num_frames.to_string())
        .arg("--fps").arg(&fps.to_string())
        .arg("--width").arg(&payload.width.unwrap_or(512).to_string())
        .arg("--height").arg(&payload.height.unwrap_or(512).to_string())
        .arg("--format").arg(format)
        .arg("--quality").arg(quality);
    
    // Add guidance scale if provided
    if let Some(guidance_scale) = payload.guidance_scale {
        command.arg("--guidance_scale").arg(&guidance_scale.to_string());
    }
    
    // Add seed if provided for reproducibility
    if let Some(seed) = payload.seed {
        command.arg("--seed").arg(&seed.to_string());
    }
    
    // Add model if provided
    if let Some(model) = &payload.model {
        command.arg("--model").arg(model);
    }
    
    // Handle new parameters from the frontend
    if let Some(interpolation_steps) = payload.interpolation_steps {
        command.arg("--interpolation_steps").arg(&interpolation_steps.to_string());
    }
    
    if let Some(animation_effect) = &payload.animation_effect {
        command.arg("--animation_effect").arg(animation_effect);
    }
    
    if let Some(transition_type) = &payload.transition_type {
        command.arg("--transition_type").arg(transition_type);
    }
    
    // Handle narration if requested
    let generate_narration = payload.narration.is_some() && payload.narration.as_ref().unwrap().enable;
    if generate_narration {
        command.arg("--narration");
        
        if let Some(narration_type) = payload.narration.as_ref().and_then(|n| n.narration_type.clone()) {
            command.arg("--narration_type").arg(narration_type);
        }
    }
    
    // Execute the command
    let status = command.status();
        
    match status {
        Ok(status) if status.success() => {
            // If successful, prepare response with paths to generated files
            let mut response_data = serde_json::json!({
                "frames": num_frames,
                "format": format,
                "quality": quality
            });
            
            // Add paths based on format
            if format == "gif" || format == "both" {
                let gif_path = format!("{}.gif", base_filename);
                let relative_gif_path = gif_path.replace("data/", "/data/");
                response_data["gif_path"] = serde_json::json!(relative_gif_path);
            }
            
            if format == "mp4" || format == "both" {
                let mp4_path = format!("{}.mp4", base_filename);
                let relative_mp4_path = mp4_path.replace("data/", "/data/");
                response_data["mp4_path"] = serde_json::json!(relative_mp4_path);
            }
            
            // Add paths to individual frames
            let frames_dir = format!("{}/frames", output_dir);
            let mut frame_paths = Vec::new();
            
            // Try to read the frames directory
            if let Ok(entries) = std::fs::read_dir(frames_dir) {
                for entry in entries.flatten() {
                    if let Some(path) = entry.path().to_str() {
                        if path.ends_with(".png") {
                            frame_paths.push(path.replace("data/", "/data/"));
                        }
                    }
                }
                frame_paths.sort();
            }
            
            response_data["frame_paths"] = serde_json::json!(frame_paths);
            response_data["total_frames"] = serde_json::json!(frame_paths.len());
            
            // Read narration if it was generated
            if generate_narration {
                let narration_path = format!("{}_narration.txt", base_filename);
                if let Ok(narration) = std::fs::read_to_string(&narration_path) {
                    response_data["narration"] = serde_json::json!(narration);
                    response_data["narration_path"] = serde_json::json!(
                        narration_path.replace("data/", "/data/")
                    );
                }
            }
            
            // Add a buffer ID for saving the content later
            response_data["buffer_id"] = serde_json::json!(dream_id);
            
            Json(response_data).into_response()
        },
        Ok(status) => {
            error!("Animation generation script failed with status: {}", status);
            (StatusCode::INTERNAL_SERVER_ERROR, "Animation generation failed").into_response()
        },
        Err(e) => {
            error!("Failed to run animation generation script: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Failed to generate animation").into_response()
        }
    }
}

// Add learning page handler
async fn learning_handler(State(state): State<AppState>) -> impl IntoResponse {
    let mut context = Context::new();
    
    // Get learning statistics if available
    if let Some(learning_system) = get_learning_system() {
        match learning_system.get_statistics() {
            Ok(stats) => {
                context.insert("stats", &stats);
                context.insert("learning_enabled", &true);
            },
            Err(err) => {
                error!("Failed to get learning statistics: {}", err);
                context.insert("error", &format!("Failed to get learning statistics: {}", err));
                context.insert("learning_enabled", &false);
            }
        }
    } else {
        context.insert("learning_enabled", &false);
    }
    
    match state.templates.render("learning.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(err) => {
            error!("Template error: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

// Add learning stats API handler
async fn learning_stats_handler() -> impl IntoResponse {
    if let Some(learning_system) = get_learning_system() {
        match learning_system.get_statistics() {
            Ok(stats) => Json(stats).into_response(),
            Err(err) => {
                error!("Failed to get learning statistics: {}", err);
                (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get learning statistics").into_response()
            }
        }
    } else {
        (StatusCode::NOT_FOUND, "Learning system not enabled").into_response()
    }
}

// Add fine-tuning handler
async fn fine_tune_handler(Json(payload): Json<FineTuneRequest>) -> impl IntoResponse {
    if let Some(learning_system) = get_learning_system() {
        // Export training data
        let training_data_path = Path::new("data/learning/training_data.jsonl");
        match learning_system.export_training_data(training_data_path) {
            Ok(count) => {
                if count < 10 {
                    return (StatusCode::BAD_REQUEST, format!("Not enough training data (only {} examples)", count)).into_response();
                }
                
                // Initialize fine-tuner
                let fine_tuner = learning::fine_tuning::FineTuner::new(
                    "data/models",
                    "data/learning/training_data.jsonl"
                );
                
                // Start fine-tuning
                match fine_tuner.start_fine_tuning(&payload.base_model, &payload.output_model) {
                    Ok(_) => {
                        info!("Started fine-tuning process");
                        Json(serde_json::json!({
                            "success": true,
                            "message": "Fine-tuning process started",
                            "examples": count
                        })).into_response()
                    },
                    Err(err) => {
                        error!("Failed to start fine-tuning: {}", err);
                        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to start fine-tuning: {}", err)).into_response()
                    }
                }
            },
            Err(err) => {
                error!("Failed to export training data: {}", err);
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to export training data: {}", err)).into_response()
            }
        }
    } else {
        (StatusCode::NOT_FOUND, "Learning system not enabled").into_response()
    }
}

// Update the feedback handler to store feedback in the learning system
async fn submit_feedback_handler(
    State(state): State<AppState>,
    Json(payload): Json<FeedbackRequest>,
) -> impl IntoResponse {
    // Get the dream data from the state
    let mut dream_data_lock = state.dream_data.lock().unwrap();
    
    if let Some(dream_data) = dream_data_lock.get_mut(&payload.dream_id) {
        // Log the feedback
        info!(
            "Received feedback for dream {}: helpful={}, comments={:?}",
            payload.dream_id, payload.helpful, payload.comments
        );
        
        // Update the dream data with the feedback
        let feedback = FeedbackInfo {
            helpful: payload.helpful,
            comments: payload.comments.clone(),
            timestamp: chrono::Utc::now(),
        };
        dream_data.feedback = Some(feedback);
        
        // Add feedback to learning system if enabled
        if let Some(learning_system) = get_learning_system() {
            let feedback_data = FeedbackData {
                dream_id: payload.dream_id.clone(),
                dream_text: dream_data.input_text.clone(),
                interpretation: dream_data.processed_data.summary.clone(),
                helpful: payload.helpful,
                comments: payload.comments.clone(),
                timestamp: chrono::Utc::now(),
                image_generated: !dream_data.image_paths.is_empty(),
                video_generated: dream_data.video_path.is_some(),
                model_used: dream_data.model_used.clone(),
            };
            
            if let Err(err) = learning_system.add_feedback(feedback_data) {
                error!("Failed to add feedback to learning system: {}", err);
            }
        }
        
        (StatusCode::OK, "Feedback received").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Dream not found").into_response()
    }
}

async fn buffer_content_handler(
    State(state): State<AppState>,
    Json(payload): Json<BufferContentRequest>,
) -> impl IntoResponse {
    // Generate a unique ID for this buffered content
    let buffer_id = Uuid::new_v4().to_string();
    
    // Set expiration time to 24 hours from now
    let now = chrono::Utc::now();
    let expires_at = now + chrono::Duration::hours(24);
    
    // Create buffered content object
    let buffered_content = BufferedContent {
        id: buffer_id.clone(),
        content_type: payload.content_type,
        data: payload.data,
        metadata: payload.metadata.unwrap_or(serde_json::json!({})),
        timestamp: now,
        expires_at,
    };
    
    // Store in buffer
    state.temp_buffer.lock().unwrap().insert(buffer_id.clone(), buffered_content);
    
    // Clean up expired content
    let mut buffer = state.temp_buffer.lock().unwrap();
    buffer.retain(|_, content| content.expires_at > now);
    
    // Return the buffer ID
    Json(serde_json::json!({
        "buffer_id": buffer_id
    })).into_response()
}

async fn save_buffered_content_handler(
    State(state): State<AppState>,
    Json(payload): Json<SaveBufferedContentRequest>,
) -> impl IntoResponse {
    let mut buffer = state.temp_buffer.lock().unwrap();
    
    match buffer.get(&payload.buffer_id) {
        Some(content) => {
            // Check if content has expired
            if content.expires_at < chrono::Utc::now() {
                buffer.remove(&payload.buffer_id);
                (StatusCode::NOT_FOUND, "Buffered content has expired").into_response()
            } else {
                // For now, just return success without actually saving
                match content.content_type.as_str() {
                    "image" => {
                        Json(serde_json::json!({
                            "success": true,
                            "file_path": format!("/data/test/example.{}", content.content_type) 
                        })).into_response()
                    },
                    "gif" => {
                        Json(serde_json::json!({
                            "success": true,
                            "file_path": format!("/data/test/example.{}", content.content_type) 
                        })).into_response()
                    },
                    "mp4" => {
                        Json(serde_json::json!({
                            "success": true,
                            "file_path": format!("/data/test/example.{}", content.content_type) 
                        })).into_response()
                    },
                    "story" => {
                        Json(serde_json::json!({
                            "success": true,
                            "file_path": format!("/data/test/example.txt") 
                        })).into_response()
                    },
                    _ => {
                        (StatusCode::BAD_REQUEST, "Unsupported content type").into_response()
                    }
                }
            }
        },
        None => (StatusCode::NOT_FOUND, "Buffered content not found").into_response()
    }
}

async fn get_buffered_content_handler(
    State(state): State<AppState>,
    axum::extract::Path(buffer_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let buffer = state.temp_buffer.lock().unwrap();
    
    if let Some(content) = buffer.get(&buffer_id) {
        // Check if content has expired
        if content.expires_at < chrono::Utc::now() {
            return StatusCode::NOT_FOUND.into_response();
        }
        
        Json(content).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

// Function to clean up expired content
fn cleanup_expired_content(temp_buffer: Arc<Mutex<HashMap<String, BufferedContent>>>) {
    let now = chrono::Utc::now();
    let mut buffer = temp_buffer.lock().unwrap();
    
    // Filter out expired content
    let expired_keys: Vec<String> = buffer
        .iter()
        .filter(|(_, content)| content.expires_at < now)
        .map(|(key, _)| key.clone())
        .collect();
    
    // Remove expired content
    for key in expired_keys {
        buffer.remove(&key);
    }
    
    info!("Cleaned up {} expired items from temporary buffer", buffer.len());
}

// Add video generation page handler
async fn video_generation_handler(State(state): State<AppState>) -> impl IntoResponse {
    let context = Context::new();
    match state.templates.render("video_generation.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(err) => {
            error!("Template error: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

// Add image generation page handler
async fn image_generation_handler(State(state): State<AppState>) -> impl IntoResponse {
    let context = Context::new();
    match state.templates.render("image_generation.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(err) => {
            error!("Template error: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn generate_enhanced_images_handler(
    Json(payload): Json<EnhancedImageGenRequest>,
) -> impl IntoResponse {
    let dream_id = Uuid::new_v4().to_string();
    let output_dir = format!("data/images/{}_enhanced", dream_id);
    
    // Create the output directory if it doesn't exist
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        error!("Failed to create output directory: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to create output directory").into_response();
    }
    
    // Call the Python script for enhanced image generation
    let mut command = Command::new("python3");
    command.arg("generate_enhanced_images.py")
        .arg("--prompt").arg(&payload.prompt)
        .arg("--output_dir").arg(&output_dir);
    
    // Add optional parameters
    if let Some(models) = &payload.models {
        command.arg("--models").arg(models);
    }
    
    if let Some(blend_mode) = &payload.blend_mode {
        command.arg("--blend_mode").arg(blend_mode);
    }
    
    if let Some(quality) = &payload.quality {
        command.arg("--quality").arg(quality);
    }
    
    if let Some(guidance_scale) = payload.guidance_scale {
        command.arg("--guidance_scale").arg(&guidance_scale.to_string());
    }
    
    if let Some(width) = payload.width {
        command.arg("--width").arg(&width.to_string());
    }
    
    if let Some(height) = payload.height {
        command.arg("--height").arg(&height.to_string());
    }
    
    if let Some(seed) = payload.seed {
        command.arg("--seed").arg(&seed.to_string());
    }
    
    if let Some(enhancement) = payload.enhancement {
        command.arg("--enhancement").arg(&enhancement.to_string());
    }
    
    if let Some(save_all) = payload.save_all {
        if save_all {
            command.arg("--save_all");
        }
    }
    
    // Execute the Python script
    let _python_status = match command.status() {
        Ok(status) if status.success() => {
            // Script executed successfully
        },
        Ok(status) => {
            error!("Python script failed with status: {}", status);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Enhanced image generation failed").into_response();
        },
        Err(e) => {
            error!("Failed to start Python script: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Enhanced image generation failed").into_response();
        }
    };
    
    // Collect the generated image paths
    let mut image_paths = Vec::new();
    let mut individual_paths = Vec::new();
    let mut intermediate_paths = Vec::new();
    
    if let Ok(entries) = std::fs::read_dir(&output_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "png" || ext == "jpg" || ext == "webp" {
                    let path_str = path.display().to_string();
                    
                    // Categorize by filename pattern
                    if path_str.contains("_intermediate_") {
                        intermediate_paths.push(path_str);
                    } else if path_str.contains("_model_") {
                        individual_paths.push(path_str);
                    } else {
                        image_paths.push(path_str);
                    }
                }
            }
        }
    }
    
    // Sort paths for consistent ordering
    image_paths.sort();
    individual_paths.sort();
    intermediate_paths.sort();
    
    // Create a buffer ID for saving the content
    let _content = BufferedContent {
        id: dream_id.clone(),
        content_type: "enhanced_image".to_string(),
        data: if !image_paths.is_empty() { image_paths[0].clone() } else { "".to_string() },
        metadata: serde_json::json!({
            "prompt": payload.prompt,
            "models": payload.models,
            "blend_mode": payload.blend_mode,
            "all_paths": image_paths,
            "individual_paths": individual_paths,
            "intermediate_paths": intermediate_paths
        }),
        timestamp: chrono::Utc::now(),
        expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
    };
    
    // Return JSON with paths
    Json(serde_json::json!({
        "success": true,
        "dream_id": dream_id,
        "message": "Enhanced images have been generated successfully",
        "images": image_paths.iter().map(|p| p.replace("data/", "/data/")).collect::<Vec<String>>(),
        "individual_images": individual_paths.iter().map(|p| p.replace("data/", "/data/")).collect::<Vec<String>>(),
        "intermediate_images": intermediate_paths.iter().map(|p| p.replace("data/", "/data/")).collect::<Vec<String>>()
    })).into_response()
}

async fn enhanced_image_generation_handler(State(state): State<AppState>) -> impl IntoResponse {
    let context = Context::new();
    match state.templates.render("enhanced_images.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(err) => {
            error!("Template error: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn upload_image_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut image_data = None;
    let mut image_name = String::new();
    
    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        if field.name().unwrap_or("") == "image" {
            image_name = field.file_name().unwrap_or("upload.jpg").to_string();
            image_data = field.bytes().await.ok();
        }
    }
    
    if let Some(data) = image_data {
        // Generate a unique ID for the uploaded image
        let image_id = Uuid::new_v4().to_string();
        let dir_path = format!("data/uploads/{}", image_id);
        let file_path = format!("{}/{}", dir_path, image_name);
        
        // Create directory for upload
        if let Err(e) = fs::create_dir_all(&dir_path).await {
            error!("Failed to create upload directory: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to save uploaded image").into_response();
        }
        
        // Save the uploaded image
        if let Err(e) = fs::write(&file_path, &data).await {
            error!("Failed to save uploaded file: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to save uploaded image").into_response();
        }
        
        // Create a buffered content entry
        let buffer_id = image_id.clone();
        let buffered_content = BufferedContent {
            id: buffer_id.clone(),
            content_type: "uploaded_image".to_string(),
            data: file_path.clone(),
            metadata: serde_json::json!({
                "original_filename": image_name,
                "size": data.len(),
                "upload_time": chrono::Utc::now().to_rfc3339()
            }),
            timestamp: chrono::Utc::now(),
            expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
        };
        
        // Store in buffer
        state.temp_buffer.lock().unwrap().insert(buffer_id.clone(), buffered_content);
        
        // Return the image ID and path
        return Json(serde_json::json!({
            "success": true,
            "image_id": image_id,
            "file_path": file_path.replace("data/", "/data/") 
        })).into_response();
    }
    
    (StatusCode::BAD_REQUEST, "No image uploaded").into_response()
}

async fn modify_image_page_handler(State(state): State<AppState>) -> impl IntoResponse {
    let context = Context::new();
    match state.templates.render("modify_image.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(err) => {
            error!("Template error: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

// Upload mask image handler
async fn upload_mask_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut image_data = None;
    let mut image_name = String::new();
    
    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        if field.name().unwrap_or("") == "mask" {
            image_name = field.file_name().unwrap_or("mask.png").to_string();
            image_data = field.bytes().await.ok();
        }
    }
    
    if let Some(data) = image_data {
        // Generate a unique ID for the uploaded mask
        let mask_id = Uuid::new_v4().to_string();
        let dir_path = format!("data/uploads/{}", mask_id);
        let file_path = format!("{}/{}", dir_path, image_name);
        
        // Create directory for upload
        if let Err(e) = fs::create_dir_all(&dir_path).await {
            error!("Failed to create mask upload directory: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to save uploaded mask").into_response();
        }
        
        // Save the uploaded mask
        if let Err(e) = fs::write(&file_path, &data).await {
            error!("Failed to save uploaded mask: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to save uploaded mask").into_response();
        }
        
        // Create a buffered content entry
        let buffer_id = mask_id.clone();
        let buffered_content = BufferedContent {
            id: buffer_id.clone(),
            content_type: "mask_image".to_string(),
            data: file_path.clone(),
            metadata: serde_json::json!({
                "original_filename": image_name,
                "size": data.len(),
                "upload_time": chrono::Utc::now().to_rfc3339()
            }),
            timestamp: chrono::Utc::now(),
            expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
        };
        
        // Store in buffer
        state.temp_buffer.lock().unwrap().insert(buffer_id.clone(), buffered_content);
        
        // Return the mask ID and path
        return Json(serde_json::json!({
            "success": true,
            "mask_id": mask_id,
            "file_path": file_path.replace("data/", "/data/") 
        })).into_response();
    }
    
    (StatusCode::BAD_REQUEST, "No mask uploaded").into_response()
}

// Inpaint page handler
async fn inpaint_image_page_handler(State(state): State<AppState>) -> impl IntoResponse {
    let context = Context::new();
    match state.templates.render("inpaint_image.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(err) => {
            error!("Template error: {}", err);
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

// Create wrapper functions that correctly drop the mutex guards before async operations
async fn simple_modify_image_handler(
    State(state): State<AppState>,
    Json(payload): Json<ModifyImageRequest>,
) -> impl IntoResponse {
    // Clone the necessary data from the buffer
    let (input_image_path, image_id) = {
        let buffer = state.temp_buffer.lock().unwrap();  // Lock mutex
        if let Some(content) = buffer.get(&payload.image_id) {
            (content.data.clone(), payload.image_id.clone())
        } else {
            drop(buffer);  // Explicitly drop the mutex guard
            return (StatusCode::NOT_FOUND, "Uploaded image not found").into_response();
        }
    };  // Mutex guard goes out of scope here
    
    // Create a unique ID for the modified image
    let output_id = Uuid::new_v4().to_string();
    let output_dir = format!("data/modified/{}", output_id);
    let output_path = format!("{}/modified.png", output_dir);
    
    // Create output directory - this is the async operation
    if let Err(e) = tokio::fs::create_dir_all(&output_dir).await {
        error!("Failed to create output directory: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to create output directory").into_response();
    }
    
    // Build command to modify image
    let mut command = Command::new("python3");
    command.arg("modify_dream_image.py")
        .arg("--input_image").arg(&input_image_path)
        .arg("--prompt").arg(&payload.prompt)
        .arg("--output_path").arg(&output_path);
    
    // Add optional parameters
    if let Some(model) = &payload.model {
        command.arg("--model").arg(model);
    }
    
    if let Some(strength) = payload.strength {
        command.arg("--strength").arg(&strength.to_string());
    }
    
    if let Some(steps) = payload.steps {
        command.arg("--steps").arg(&steps.to_string());
    }
    
    if let Some(guidance_scale) = payload.guidance_scale {
        command.arg("--guidance_scale").arg(&guidance_scale.to_string());
    }
    
    if let Some(seed) = payload.seed {
        command.arg("--seed").arg(&seed.to_string());
    }
    
    if let Some(enhancement) = payload.enhancement {
        command.arg("--enhancement").arg(&enhancement.to_string());
    }
    
    // Execute the image modification
    let python_status = match command.status() {
        Ok(status) => status,
        Err(e) => {
            error!("Failed to start image modification script: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to start image modification").into_response();
        }
    };
    
    if !python_status.success() {
        error!("Image modification script failed with status: {}", python_status);
        return (StatusCode::INTERNAL_SERVER_ERROR, "Image modification failed").into_response();
    }
    
    // Create the buffer result
    let buffered_result = BufferedContent {
        id: output_id.clone(),
        content_type: "modified_image".to_string(),
        data: output_path.clone(),
        metadata: serde_json::json!({
            "original_image_id": image_id,
            "prompt": payload.prompt,
            "model": payload.model,
            "strength": payload.strength,
            "modification_time": chrono::Utc::now().to_rfc3339()
        }),
        timestamp: chrono::Utc::now(),
        expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
    };
    
    // Store the result in the buffer with a new lock
    {
        let mut buffer = state.temp_buffer.lock().unwrap();
        buffer.insert(output_id.clone(), buffered_result);
    }
    
    // Return success with the image paths
    Json(serde_json::json!({
        "success": true,
        "original_image": input_image_path.replace("data/", "/data/"),
        "modified_image": output_path.replace("data/", "/data/"),
        "buffer_id": output_id
    })).into_response()
}

// Simple inpaint handler
async fn simple_inpaint_image_handler(
    State(state): State<AppState>,
    Json(payload): Json<InpaintImageRequest>,
) -> impl IntoResponse {
    // Get the image and mask data within a scope that drops the mutex guard
    let (image_data, mask_data, image_id, mask_id) = {
        let buffer = state.temp_buffer.lock().unwrap();
        
        let image_content = match buffer.get(&payload.image_id) {
            Some(content) => content.clone(),
            None => {
                drop(buffer);
                return (StatusCode::NOT_FOUND, "Uploaded image not found").into_response();
            }
        };
        
        let mask_content = match buffer.get(&payload.mask_id) {
            Some(content) => content.clone(),
            None => {
                drop(buffer);
                return (StatusCode::NOT_FOUND, "Uploaded mask not found").into_response();
            }
        };
        
        (image_content.data.clone(), mask_content.data.clone(), 
         payload.image_id.clone(), payload.mask_id.clone())
    }; // Mutex guard is dropped here
    
    // Create a unique ID for the inpainted image
    let output_id = Uuid::new_v4().to_string();
    let output_dir = format!("data/inpainted/{}", output_id);
    let output_path = format!("{}/inpainted.png", output_dir);
    
    // Create output directory - this is the async operation
    if let Err(e) = tokio::fs::create_dir_all(&output_dir).await {
        error!("Failed to create output directory: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to create output directory").into_response();
    }
    
    // Build command to inpaint image
    let mut command = Command::new("python3");
    command.arg("inpaint_dream_image.py")
        .arg("--input_image").arg(&image_data)
        .arg("--mask_image").arg(&mask_data)
        .arg("--prompt").arg(&payload.prompt)
        .arg("--output_path").arg(&output_path);
    
    // Add optional parameters
    if let Some(model) = &payload.model {
        command.arg("--model").arg(model);
    }
    
    if let Some(steps) = payload.steps {
        command.arg("--steps").arg(&steps.to_string());
    }
    
    if let Some(guidance_scale) = payload.guidance_scale {
        command.arg("--guidance_scale").arg(&guidance_scale.to_string());
    }
    
    if let Some(seed) = payload.seed {
        command.arg("--seed").arg(&seed.to_string());
    }
    
    if let Some(enhancement) = payload.enhancement {
        command.arg("--enhancement").arg(&enhancement.to_string());
    }
    
    // Execute the inpainting
    let python_status = match command.status() {
        Ok(status) => status,
        Err(e) => {
            error!("Failed to start inpainting script: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to start inpainting").into_response();
        }
    };
    
    if !python_status.success() {
        error!("Inpainting script failed with status: {}", python_status);
        return (StatusCode::INTERNAL_SERVER_ERROR, "Inpainting failed").into_response();
    }
    
    // Create the buffer content
    let buffered_result = BufferedContent {
        id: output_id.clone(),
        content_type: "inpainted_image".to_string(),
        data: output_path.clone(),
        metadata: serde_json::json!({
            "original_image_id": image_id,
            "mask_id": mask_id,
            "prompt": payload.prompt,
            "model": payload.model,
            "inpainting_time": chrono::Utc::now().to_rfc3339()
        }),
        timestamp: chrono::Utc::now(),
        expires_at: chrono::Utc::now() + chrono::Duration::hours(24),
    };
    
    // Store the result in the buffer with a new lock
    {
        let mut buffer = state.temp_buffer.lock().unwrap();
        buffer.insert(output_id.clone(), buffered_result);
    }
    
    // Return success with the image paths
    Json(serde_json::json!({
        "success": true,
        "original_image": image_data.replace("data/", "/data/"),
        "mask_image": mask_data.replace("data/", "/data/"),
        "inpainted_image": output_path.replace("data/", "/data/"),
        "buffer_id": output_id
    })).into_response()
} 