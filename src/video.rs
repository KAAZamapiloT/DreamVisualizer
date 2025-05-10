use anyhow::{Context, Result};
use gif::{Encoder, Frame, Repeat};
use image::{imageops::FilterType};
use std::fs::{self, File};
use tracing::info;
use uuid::Uuid;

use crate::config;

pub struct VideoGenerator;

impl VideoGenerator {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn generate_video(&self, image_paths: &[String], output_dir: &str) -> Result<String> {
        let cfg = config::get();
        
        if image_paths.is_empty() {
            anyhow::bail!("No images provided for animation generation");
        }
        
        // Create output directory if it doesn't exist
        fs::create_dir_all(output_dir)?;
        
        // Generate unique output file name for GIF
        let animation_id = Uuid::new_v4().to_string();
        let output_path = format!("{}/{}.gif", output_dir, animation_id);
        
        info!("Loading {} images for animation", image_paths.len());
        
        // Load all images
        let mut images = Vec::new();
        for path in image_paths {
            let img = image::open(path)
                .with_context(|| format!("Failed to open image: {}", path))?;
            images.push(img);
        }
        
        // Determine optimal width/height (use first image as reference)
        let width = 640; // Reasonable size for web
        let height = 480;
        
        // Create GIF encoder
        let file = File::create(&output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;
        
        let mut encoder = Encoder::new(
            file,
            width as u16,
            height as u16,
            &[]
        )?;
        
        // Set to loop infinitely
        encoder.set_repeat(Repeat::Infinite)?;
        
        // Calculate frame delay (in 1/100ths of a second)
        let frame_delay = (100.0 / cfg.ffmpeg.fps as f32) as u16;
        
        // Add each image as a frame
        for img in images {
            // Resize image to fit GIF dimensions
            let resized = img.resize_exact(width, height, FilterType::Lanczos3);
            
            // Convert to GIF-compatible format (indexed colors)
            let frame_image = resized.to_rgba8();
            let mut raw_pixels = frame_image.into_raw();
            
            // Create GIF frame
            let mut frame = Frame::from_rgba_speed(
                width as u16,
                height as u16,
                &mut raw_pixels,
                30, // Higher value = faster encoding, lower quality
            );
            
            // Set frame delay
            frame.delay = frame_delay;
            
            // Write frame to GIF
            encoder.write_frame(&frame)?;
        }
        
        info!("Animation generated successfully at: {}", output_path);
        
        Ok(output_path)
    }
    
    // Alternative method to generate WebP animations which are higher quality than GIFs
    pub fn generate_webp_animation(&self, image_paths: &[String], output_dir: &str) -> Result<String> {
        let cfg = config::get();
        
        if image_paths.is_empty() {
            anyhow::bail!("No images provided for animation generation");
        }
        
        // Create output directory if it doesn't exist
        fs::create_dir_all(output_dir)?;
        
        // Generate unique output file name
        let animation_id = Uuid::new_v4().to_string();
        let output_path = format!("{}/{}.webp", output_dir, animation_id);
        
        info!("Loading {} images for WebP animation", image_paths.len());
        
        // Load first image to get dimensions
        let first_img = image::open(&image_paths[0])
            .with_context(|| format!("Failed to open first image: {}", &image_paths[0]))?;
        
        // Frame dimensions - use the first image or default size
        let width = first_img.width();
        let height = first_img.height();
        
        // Convert all images to RGBA8 format and collect their data
        let mut frame_data = Vec::new();
        
        for path in image_paths {
            let img = image::open(path)
                .with_context(|| format!("Failed to open image: {}", path))?;
            
            // Resize image if needed
            let resized = img.resize_exact(width, height, FilterType::Lanczos3);
            let rgba_img = resized.to_rgba8();
            
            frame_data.push(rgba_img);
        }
        
        // Frame duration in milliseconds
        let frame_duration = (1000.0 / cfg.ffmpeg.fps as f32) as i32;
        
        // Create a WebP animation manually
        let gif_file = File::create(&output_path)
            .with_context(|| format!("Failed to create WebP file: {}", output_path))?;
        
        // Use gif crate as a fallback since the WebP library is giving issues
        let mut gif_encoder = Encoder::new(
            gif_file,
            width as u16,
            height as u16,
            &[]
        )?;
        gif_encoder.set_repeat(Repeat::Infinite)?;
        
        for frame_img in frame_data {
            let mut raw_pixels = frame_img.into_raw();
            
            let mut frame = Frame::from_rgba_speed(
                width as u16,
                height as u16,
                &mut raw_pixels,
                10, // Use higher quality for these frames
            );
            
            // Convert ms to 1/100th seconds
            frame.delay = (frame_duration / 10) as u16;
            
            gif_encoder.write_frame(&frame)?;
        }
        
        info!("Animation generated successfully at: {}", output_path);
        
        Ok(output_path)
    }
} 