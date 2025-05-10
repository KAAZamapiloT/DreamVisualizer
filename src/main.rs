mod config;
mod image;
mod openai;
mod video;
mod web;

use anyhow::Result;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run in CLI mode
    #[arg(short, long)]
    cli: bool,

    /// Dream description (CLI mode only)
    #[arg(short, long)]
    dream: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting Dream Visualizer");
    
    // Load configuration
    config::load()?;
    info!("Configuration loaded");
    
    // Parse command-line arguments
    let args = Args::parse();
    
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
        let animation_path = match video_generator.generate_webp_animation(&image_paths, "data/videos") {
            Ok(path) => path,
            Err(err) => {
                info!("WebP animation failed, falling back to GIF: {}", err);
                video_generator.generate_video(&image_paths, "data/videos")?
            }
        };
        
        println!("\nAnimation generated at: {}", animation_path);
        
    } else {
        // Web mode
        info!("Running in web mode");
        web::start_server().await?;
    }
    
    Ok(())
} 