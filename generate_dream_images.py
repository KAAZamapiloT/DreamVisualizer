import argparse
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import random
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate dream images using Stable Diffusion on GPU.")
    parser.add_argument('--prompt', type=str, required=True, help='Dream description prompt')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--num_images', type=int, default=4, help='Number of images to generate')
    parser.add_argument('--model', type=str, default='stabilityai/stable-diffusion-2-1', help='HuggingFace model name')
    parser.add_argument('--quality', type=str, default='high', choices=['standard', 'high', 'ultra'], 
                        help='Quality level for generation')
    parser.add_argument('--guidance_scale', type=float, default=7.5, 
                        help='Guidance scale (higher values follow prompt more closely)')
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(args.seed)
    else:
        generator = None

    # Set quality parameters
    if args.quality == 'ultra':
        num_inference_steps = 100
    elif args.quality == 'high':
        num_inference_steps = 50
    else:  # standard
        num_inference_steps = 30

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None  # Disable safety checker for creative freedom
    )
    pipe = pipe.to(device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(args.num_images):
        print(f"Generating image {i+1}/{args.num_images}...")
        
        # Generate image with quality settings
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]
        
        image_path = Path(args.output_dir) / f"dream_{i+1:02d}.png"
        image.save(image_path)
        print(f"Saved: {image_path}")

    print("All images generated!")

if __name__ == "__main__":
    main() 