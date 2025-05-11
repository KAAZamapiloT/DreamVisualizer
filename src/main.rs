mod config;
mod image;
mod openai;
mod video;
mod web;
mod learning;

use anyhow::Result;
use clap::Parser;
use tracing::{info, error};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run in CLI mode instead of web server
    #[arg(short, long)]
    cli: bool,
    
    /// Dream description (for CLI mode)
    #[arg(short, long)]
    dream: Option<String>,
    
    /// Enable AI learning from feedback
    #[arg(long, default_value_t = true)]
    enable_learning: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting Dream Visualizer");
    
    // Load configuration
    config::load()?;
    info!("Configuration loaded");
    
    // Parse command line arguments
    let args = Args::parse();
    
    // Initialize learning system if enabled
    if args.enable_learning {
        info!("AI learning system enabled");
        let learning_system = learning::LearningSystem::new("data/learning")?;
        
        // Store learning system in a global state (in a real app, this would be handled better)
        web::init_learning_system(learning_system);
    } else {
        info!("AI learning system disabled");
    }
    
    if args.cli {
        // CLI mode
        info!("Running in CLI mode");
        
        // Get dream description
        let dream_text = match args.dream {
            Some(text) => text,
            None => {
                println!("Please enter your dream description:");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                input
            }
        };
        
        // Process the dream with OpenAI
        info!("Processing dream description with OpenAI");
        let processed_data = openai::process_dream_description(&dream_text).await?;
        
        println!("\nDream Interpretation:");
        println!("Title: {}", processed_data.title);
        println!("Summary: {}", processed_data.summary);
        println!("Mood: {}", processed_data.mood);
        println!("Style: {}", processed_data.style_suggestion);
        
        println!("\nVisual Elements:");
        for element in &processed_data.elements {
            println!("- {} (importance: {}): {}", element.name, element.importance, element.description);
        }
        
        println!("\nColor Palette:");
        for color in &processed_data.color_palette {
            println!("- {}", color);
        }
        
        // Generate images
        info!("Generating images");
        let image_generator = image::ImageGenerator::new("data/images")?;
        let image_paths = image_generator.generate_images(&processed_data).await?;
        
        println!("\nGenerated Images:");
        for path in &image_paths {
            println!("- {}", path);
        }
        
        // Generate video
        info!("Generating animation");
        let video_generator = video::VideoGenerator::new();
        
        // Try WebP first (higher quality), fall back to GIF if it fails
        match video_generator.generate_webp_animation(&image_paths, "data/videos") {
            Ok(path) => {
                println!("\nGenerated Animation: {}", path);
            },
            Err(err) => {
                error!("Failed to generate WebP animation: {}", err);
                // Fall back to GIF
                match video_generator.generate_video(&image_paths, "data/videos") {
                    Ok(path) => {
                        println!("\nGenerated Animation: {}", path);
                    },
                    Err(err) => {
                        error!("Failed to generate GIF animation: {}", err);
                    }
                }
            }
        }
        
        Ok(())
    } else {
        // Web server mode
        info!("Running in web mode");
        web::start_server().await
    }
} 