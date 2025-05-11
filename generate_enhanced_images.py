import argparse
import torch
import os
import gc
from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageChops
import os
from tqdm import tqdm

# Set CUDA memory allocation configuration for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Available models for blending
MODEL_OPTIONS = {
    'sd21': 'stabilityai/stable-diffusion-2-1',
    'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
    'sd15': 'runwayml/stable-diffusion-v1-5',
    'dreamshaper': 'dreamshaper/dreamshaper-8',
    'openjourney': 'prompthero/openjourney',
    'realistic': 'dreamlike-art/dreamlike-photoreal-2.0',
    'deliberate': 'XpucT/Deliberate'
}

# Blending modes
BLEND_MODES = ['soft_light', 'hard_light', 'overlay', 'screen', 'multiply', 'difference', 'luminosity']

def blend_images(image1, image2, mode='soft_light', alpha=0.5):
    """Blend two images using different blending modes."""
    if mode == 'soft_light':
        # Soft light blending
        result = ImageChops.soft_light(image1, image2)
    elif mode == 'hard_light':
        # Hard light blending
        result = ImageChops.hard_light(image1, image2)
    elif mode == 'overlay':
        # Overlay blending (similar to soft light but more contrast)
        result = ImageChops.overlay(image1, image2)
    elif mode == 'screen':
        # Screen blending (lightens image)
        result = ImageChops.screen(image1, image2)
    elif mode == 'multiply':
        # Multiply blending (darkens image)
        result = ImageChops.multiply(image1, image2)
    elif mode == 'difference':
        # Difference blending (creates unique colors)
        result = ImageChops.difference(image1, image2)
    elif mode == 'add':
        # Additive blending
        result = ImageChops.add(image1, image2, scale=2.0)
    elif mode == 'luminosity':
        # Convert to RGBA if not already
        if image1.mode != 'RGBA':
            image1 = image1.convert('RGBA')
        if image2.mode != 'RGBA':
            image2 = image2.convert('RGBA')
            
        # Extract luminosity from image2 and apply to image1's colors
        r1, g1, b1, a1 = image1.split()
        r2, g2, b2, a2 = image2.split()
        
        # Calculate luminosity from image2
        luminosity = Image.new('L', image1.size)
        for x in range(image1.width):
            for y in range(image1.height):
                r, g, b = image2.getpixel((x, y))[:3]
                # Standard luminosity formula: 0.21*R + 0.72*G + 0.07*B
                lum = int(0.21 * r + 0.72 * g + 0.07 * b)
                luminosity.putpixel((x, y), lum)
        
        # Create new image with image1's colors but image2's luminosity
        result = Image.merge('RGBA', (r1, g1, b1, luminosity))
    else:
        # Default to alpha blend
        result = Image.blend(image1, image2, alpha)
    
    return result

def enhance_image(image, enhancement_level=1.2):
    """Apply various enhancements to improve image quality."""
    # Increase contrast slightly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(enhancement_level)
    
    # Increase color saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(enhancement_level)
    
    # Apply subtle sharpening
    image = image.filter(ImageFilter.SHARPEN)
    
    return image

def clear_gpu_memory():
    """Free up CUDA memory to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def generate_with_model(model_name, prompt, height, width, steps, guidance_scale, seed=None):
    """Generate an image using the specified model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Initialize appropriate model based on name
        if 'xl' in model_name.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                safety_checker=None,
                use_safetensors=True,
                variant="fp16" if device == 'cuda' else None
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                safety_checker=None
            )
        
        pipe = pipe.to(device)
        
        # Set up generator for reproducibility if seed provided
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None
        
        # Generate image with attention to memory usage
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        # Clear GPU memory
        del pipe
        clear_gpu_memory()
        
        return image
        
    except torch.cuda.OutOfMemoryError:
        # Handle CUDA out of memory errors gracefully
        clear_gpu_memory()
        print(f"ERROR: CUDA out of memory when processing model {model_name}.")
        print("Try reducing image size, using fewer models, or setting lower quality.")
        if device == 'cuda':
            # Return a placeholder error image
            error_img = Image.new('RGB', (width, height), color='red')
            return error_img
        
    except Exception as e:
        # Handle other errors
        clear_gpu_memory()
        print(f"Error generating with model {model_name}: {str(e)}")
        # Return a placeholder error image with different color
        error_img = Image.new('RGB', (width, height), color='yellow')
        return error_img

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced dream images using multiple models and blending.")
    parser.add_argument('--prompt', type=str, required=True, help='Dream description prompt')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate with each blend')
    parser.add_argument('--models', type=str, default='sd21,sdxl', help='Comma-separated list of models to use: sd21,sdxl,sd15,dreamshaper,openjourney,realistic,deliberate')
    parser.add_argument('--blend_mode', type=str, default='soft_light', choices=BLEND_MODES, help='Blending mode for combining model outputs')
    parser.add_argument('--quality', type=str, default='high', choices=['standard', 'high', 'ultra'], help='Quality level')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale (higher = follows prompt more closely)')
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--enhancement', type=float, default=1.2, help='Enhancement level for final image')
    parser.add_argument('--save_all', action='store_true', help='Also save individual model outputs')
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Individual model output directory
    if args.save_all:
        individual_dir = output_dir / "individual_models"
        individual_dir.mkdir(exist_ok=True)
    
    # Set quality parameters
    if args.quality == 'ultra':
        num_inference_steps = 100
    elif args.quality == 'high':
        num_inference_steps = 50
    else:  # standard
        num_inference_steps = 30
    
    # Parse models to use
    selected_models = [m.strip() for m in args.models.split(',')]
    model_names = []
    for model_key in selected_models:
        if model_key in MODEL_OPTIONS:
            model_names.append(MODEL_OPTIONS[model_key])
        else:
            # Assume direct model path
            model_names.append(model_key)
    
    if len(model_names) < 2:
        print("Warning: Blending requires at least 2 models. Adding SD 2.1 as second model.")
        model_names.append(MODEL_OPTIONS['sd21'])
    
    print(f"Using models: {model_names}")
    print(f"Blend mode: {args.blend_mode}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Check CUDA memory constraints
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_mb = free_memory / (1024 * 1024)
        print(f"Available CUDA memory: {free_memory_mb:.2f} MB")
        
        # Check if likely to run out of memory
        memory_per_model_estimate = 2500  # Rough estimate for SD models in MB
        if 'xl' in args.models:
            memory_per_model_estimate = 6000  # SDXL needs more memory
            
        if free_memory_mb < memory_per_model_estimate and len(model_names) > 1:
            print(f"WARNING: You may not have enough CUDA memory ({free_memory_mb:.2f} MB) to run multiple models.")
            print("Consider reducing image size, using fewer models, or switching to 'standard' quality.")
    
    # Generate images
    for i in range(args.num_images):
        # Use different seed for each image if no seed specified
        if args.seed is None:
            current_seed = random.randint(0, 2147483647)
        else:
            current_seed = args.seed + i
        
        print(f"\nGenerating image set {i+1}/{args.num_images} with seed {current_seed}...")
        
        # Generate images from each model
        model_images = []
        for idx, model in enumerate(model_names):
            print(f"Generating with model {idx+1}/{len(model_names)}: {model}")
            image = generate_with_model(
                model,
                args.prompt,
                args.height,
                args.width,
                num_inference_steps,
                args.guidance_scale,
                current_seed
            )
            model_images.append(image)
            
            # Save individual model output if requested
            if args.save_all:
                model_short_name = model.split('/')[-1]
                image_path = individual_dir / f"dream_{i+1:02d}_{model_short_name}.png"
                image.save(image_path)
                print(f"Saved individual model output: {image_path}")
            
            # Force memory cleanup after each model generation
            clear_gpu_memory()
        
        # Blend images from different models
        print(f"Blending images using {args.blend_mode}...")
        blended_image = model_images[0]
        for idx, image in enumerate(model_images[1:], 1):
            blended_image = blend_images(blended_image, image, mode=args.blend_mode)
            
            # Save intermediate blend if requested
            if args.save_all and idx < len(model_images) - 1:
                intermediate_path = output_dir / f"dream_{i+1:02d}_blend_step_{idx}.png"
                blended_image.save(intermediate_path)
        
        # Enhance the final blended image
        enhanced_image = enhance_image(blended_image, args.enhancement)
        
        # Save final output
        final_path = output_dir / f"dream_{i+1:02d}_enhanced.png"
        enhanced_image.save(final_path)
        print(f"Saved enhanced image: {final_path}")

    print("\nAll images generated successfully!")

if __name__ == "__main__":
    main() 