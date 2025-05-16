import argparse
import torch
import os
import gc
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

# Set CUDA memory allocation configuration for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Available models for image inpainting
MODEL_OPTIONS = {
    'sd21-inpaint': 'stabilityai/stable-diffusion-2-inpainting',
    'sd15-inpaint': 'runwayml/stable-diffusion-inpainting',
    'dreamshaper-inpaint': 'Lykon/dreamshaper-7-inpainting'
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
    
    return image

def inpaint_image(input_image_path, mask_image_path, prompt, model_name='sd21-inpaint', 
                  guidance_scale=7.5, steps=30, seed=None, enhancement=1.2):
    """Edit a specific part of an image based on the mask and prompt."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load the input image and mask
        init_image = Image.open(input_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("L")  # Grayscale
        
        # Ensure the mask and image have the same dimensions
        if init_image.size != mask_image.size:
            mask_image = mask_image.resize(init_image.size)
        
        # Invert the mask if needed (white = inpaint area)
        # In the UI, we'll ask users to paint over areas they want to change
        inverted_mask = ImageOps.invert(mask_image)
        
        # Resize to a size compatible with the model (must be divisible by 8)
        width, height = init_image.size
        new_width, new_height = width, height
        
        # Ensure dimensions are divisible by 8 (required by the model)
        if width % 8 != 0:
            new_width = (width // 8) * 8
        if height % 8 != 0:
            new_height = (height // 8) * 8
        
        # Check if we need to resize
        max_size = 768  # Maximum size limit
        if new_width > max_size or new_height > max_size or width != new_width or height != new_height:
            if width > height:
                ratio = max_size / width if width > max_size else 1
                new_width = min(max_size, new_width)
                new_height = int(height * ratio)
                # Make sure height is divisible by 8
                new_height = (new_height // 8) * 8
            else:
                ratio = max_size / height if height > max_size else 1
                new_height = min(max_size, new_height)
                new_width = int(width * ratio)
                # Make sure width is divisible by 8
                new_width = (new_width // 8) * 8
            
            init_image = init_image.resize((new_width, new_height))
            inverted_mask = inverted_mask.resize((new_width, new_height))
            print(f"Resized image to {new_width}x{new_height}")
        
        # Get the model path
        if model_name in MODEL_OPTIONS:
            model_path = MODEL_OPTIONS[model_name]
        else:
            model_path = model_name  # Assume direct model path
            
        # Initialize the inpainting pipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
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
        
        # Perform the inpainting
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                image=init_image,
                mask_image=inverted_mask,  # The inverted mask defines the area to inpaint
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator
            )
        
        # Get the output image
        inpainted_image = output.images[0]
        
        # Apply enhancements
        if enhancement > 1.0:
            inpainted_image = enhance_image(inpainted_image, enhancement)
        
        # Clear GPU memory
        del pipe
        clear_gpu_memory()
        
        return inpainted_image, None
        
    except torch.cuda.OutOfMemoryError:
        # Handle CUDA out of memory errors gracefully
        clear_gpu_memory()
        error_msg = "CUDA out of memory. Try using a smaller image or reducing quality settings."
        print(f"ERROR: {error_msg}")
        return None, error_msg
        
    except Exception as e:
        # Handle other errors
        clear_gpu_memory()
        error_msg = f"Error inpainting image: {str(e)}"
        print(f"ERROR: {error_msg}")
        return None, error_msg

def main():
    parser = argparse.ArgumentParser(description="Edit specific parts of an image using AI inpainting.")
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--mask_image', type=str, required=True, help='Path to the mask image (white areas will be replaced)')
    parser.add_argument('--prompt', type=str, required=True, help='Description for the inpainted area')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the edited image')
    parser.add_argument('--model', type=str, default='sd21-inpaint', help='Model to use: sd21-inpaint, sd15-inpaint, dreamshaper-inpaint')
    parser.add_argument('--steps', type=int, default=30, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--enhancement', type=float, default=1.2, help='Enhancement level for final image')
    
    args = parser.parse_args()
    
    # Check CUDA memory constraints
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_memory_mb = free_memory / (1024 * 1024)
        print(f"Available CUDA memory: {free_memory_mb:.2f} MB")
        
        if free_memory_mb < 4000:  # Inpainting typically needs at least ~4GB VRAM
            print(f"WARNING: Low CUDA memory ({free_memory_mb:.2f} MB) may cause issues.")
            print("Consider reducing image size or using lower quality settings.")
    
    # Inpaint the image
    inpainted_image, error = inpaint_image(
        args.input_image,
        args.mask_image,
        args.prompt,
        model_name=args.model,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=args.seed,
        enhancement=args.enhancement
    )
    
    if inpainted_image:
        # Save the inpainted image
        inpainted_image.save(args.output_path)
        print(f"Inpainted image saved to: {args.output_path}")
    else:
        print(f"Failed to inpaint image: {error}")

if __name__ == "__main__":
    main() 