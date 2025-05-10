use anyhow::Result;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{Html, IntoResponse, Redirect},
    routing::{get, post},
    Router,
};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tera::{Context, Tera};
use tokio::fs;
use tower_http::services::ServeDir;
use tracing::{error, info};
use uuid::Uuid;

use crate::config;
use crate::image::ImageGenerator;
use crate::openai::{self, DreamSceneData};
use crate::video::VideoGenerator;

#[derive(Clone)]
pub struct AppState {
    templates: Tera,
    dream_data: Arc<Mutex<HashMap<String, DreamData>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DreamData {
    id: String,
    input_text: String,
    processed_data: DreamSceneData,
    video_path: Option<String>,
    image_paths: Vec<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
}

pub async fn start_server() -> Result<()> {
    let cfg = config::get();
    
    // Set up Tera templates
    let mut tera = Tera::default();
    tera.add_raw_templates(vec![
        ("index.html", include_str!("../templates/index.html")),
        ("result.html", include_str!("../templates/result.html")),
        ("history.html", include_str!("../templates/history.html")),
    ])?;
    
    // Create directories for storing generated content
    let data_dir = Path::new("data");
    let images_dir = data_dir.join("images");
    let videos_dir = data_dir.join("videos");
    
    fs::create_dir_all(&images_dir).await?;
    fs::create_dir_all(&videos_dir).await?;
    
    // Set up app state
    let app_state = AppState {
        templates: tera,
        dream_data: Arc::new(Mutex::new(HashMap::new())),
    };
    
    // Build our application with routes
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/process", post(process_handler))
        .route("/result/:id", get(result_handler))
        .route("/history", get(history_handler))
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
    };
    
    state.dream_data.lock().unwrap().insert(dream_id.clone(), dream_data);
    
    // Redirect to the result page
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