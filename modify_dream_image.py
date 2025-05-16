import argparse
import torch
import os
import gc
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageEnhance
import numpy as np
import random

# Set CUDA memory allocation configuration for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Available models for image modification
MODEL_OPTIONS = {
    'sd21': 'stabilityai/stable-diffusion-2-1',
    'sd15': 'runwayml/stable-diffusion-v1-5',
    'dreamshaper': 'dreamshaper/dreamshaper-8',
    'openjourney': 'prompthero/openjourney',
    'realistic': 'dreamlike-art/dreamlike-photoreal-2.0'
}

def clear_gpu_memory():
    """Free up CUDA memory to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def enhance_image(image, enhancement_level=1.2):
    """Apply various enhancements to improve image quality."""
    # Increase contrast slightly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(enhancement_level)
    
    # Increase color saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(enhancement_level)
    
    # Increase brightness slightly
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    
    return image

def modify_image(input_image_path, prompt, model_name='sd21', strength=0.75, 
                 guidance_scale=7.5, steps=50, seed=None, enhancement=1.2):
    """Modify an input image based on the prompt using img2img pipeline."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load the input image
        init_image = Image.open(input_image_path).convert("RGB")
        
        # Resize to a size compatible with the model
        width, height = init_image.size
        max_size = 768
        
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            init_image = init_image.resize((new_width, new_height))
            print(f"Resized image to {new_width}x{new_height}")
        
        # Get the model path
        if model_name in MODEL_OPTIONS:
            model_path = MODEL_OPTIONS[model_name]
        else:
            model_path = model_name  # Assume direct model path
            
        # Initialize the img2img pipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            safety_checker=None
        )
        
        pipe = pipe.to(device)
        
        # Set up generator for reproducibility if seed provided
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(random.randint(0, 2147483647))
        
        # Perform the image modification
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator
            )
        
        # Get the output image
        modified_image = output.images[0]
        
        # Apply enhancements
        if enhancement > 1.0:
            modified_image = enhance_image(modified_image, enhancement)
        
        # Clear GPU memory
        del pipe
        clear_gpu_memory()
        
        return modified_image, None
        
    except torch.cuda.OutOfMemoryError:
        # Handle CUDA out of memory errors gracefully
        clear_gpu_memory()
        error_msg = "CUDA out of memory. Try using a smaller image or reducing quality settings."
        print(f"ERROR: {error_msg}")
        return None, error_msg
        
    except Exception as e:
        # Handle other errors
        clear_gpu_memory()
        error_msg = f"Error modifying image: {str(e)}"
        print(f"ERROR: {error_msg}")
        return None, error_msg

def main():
    parser = argparse.ArgumentParser(description="Modify an existing image based on a dream description.")
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='Dream description prompt for modification')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the modified image')
    parser.add_argument('--model', type=str, default='sd21', help='Model to use: sd21, sd15, dreamshaper, etc.')
    parser.add_argument('--strength', type=float, default=0.75, help='Strength of modification (0.0-1.0)')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--enhancement', type=float, default=1.2, help='Enhancement level for final image')
    
    args = parser.parse_args()
    
    # Check CUDA memory constraints
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_mb = free_memory / (1024 * 1024)
        print(f"Available CUDA memory: {free_memory_mb:.2f} MB")
        
        if free_memory_mb < 4000:  # Img2Img typically needs at least ~4GB VRAM
            print(f"WARNING: Low CUDA memory ({free_memory_mb:.2f} MB) may cause issues.")
            print("Consider reducing image size or using lower quality settings.")
    
    # Modify the image
    modified_image, error = modify_image(
        args.input_image,
        args.prompt,
        model_name=args.model,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=args.seed,
        enhancement=args.enhancement
    )
    
    if modified_image:
        # Save the modified image
        modified_image.save(args.output_path)
        print(f"Modified image saved to: {args.output_path}")
    else:
        print(f"Failed to modify image: {error}")

if __name__ == "__main__":
    main() 