import argparse
import os
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageOps
import numpy as np
import random
from tqdm import tqdm
import subprocess
import shutil
import re  # Add import for regex

def apply_animation_effect(images, effect='none', **kwargs):
    """Apply animation effects to the frames."""
    if effect == 'none' or not effect:
        return images
    
    processed_images = []
    num_images = len(images)
    
    if effect == 'zoom-in':
        # Zoom in effect (start wide, end closer)
        for i, img in enumerate(images):
            # Calculate zoom factor based on position in sequence
            zoom_factor = 1.0 + (i / (num_images - 1)) * 0.3 if num_images > 1 else 1.0
            w, h = img.size
            
            # Calculate new size with zoom
            new_w = int(w / zoom_factor)
            new_h = int(h / zoom_factor)
            
            # Crop from center
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            right = left + new_w
            bottom = top + new_h
            
            cropped = img.crop((left, top, right, bottom))
            resized = cropped.resize((w, h), Image.Resampling.LANCZOS)
            processed_images.append(resized)
    
    elif effect == 'zoom-out':
        # Zoom out effect (start close, end wide)
        for i, img in enumerate(images):
            # Calculate zoom factor based on position in sequence (reverse of zoom-in)
            zoom_factor = 1.0 + ((num_images - 1 - i) / (num_images - 1)) * 0.3 if num_images > 1 else 1.0
            w, h = img.size
            
            # Calculate new size with zoom
            new_w = int(w / zoom_factor)
            new_h = int(h / zoom_factor)
            
            # Crop from center
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            right = left + new_w
            bottom = top + new_h
            
            cropped = img.crop((left, top, right, bottom))
            resized = cropped.resize((w, h), Image.Resampling.LANCZOS)
            processed_images.append(resized)
    
    elif effect == 'pan-left':
        # Pan from right to left
        for i, img in enumerate(images):
            w, h = img.size
            pan_amount = int((1 - i / (num_images - 1)) * w * 0.3) if num_images > 1 else 0
            
            # Create a wider image with black padding
            wider = Image.new('RGB', (w + int(w * 0.3), h), (0, 0, 0))
            wider.paste(img, (pan_amount, 0))
            
            # Crop back to original size
            cropped = wider.crop((0, 0, w, h))
            processed_images.append(cropped)
    
    elif effect == 'pan-right':
        # Pan from left to right
        for i, img in enumerate(images):
            w, h = img.size
            pan_amount = int((i / (num_images - 1)) * w * 0.3) if num_images > 1 else 0
            
            # Create a wider image with black padding
            wider = Image.new('RGB', (w + int(w * 0.3), h), (0, 0, 0))
            wider.paste(img, (0, 0))
            
            # Crop back to original size
            cropped = wider.crop((pan_amount, 0, pan_amount + w, h))
            processed_images.append(cropped)
    
    elif effect == 'dissolve':
        # Simple dissolve effect
        for i in range(num_images):
            if i == 0:
                processed_images.append(images[i])
            else:
                # Blend with previous image
                alpha = 0.7  # Strength of dissolve effect
                blended = Image.blend(images[i-1], images[i], alpha)
                processed_images.append(blended)
    
    elif effect == 'morph':
        # Apply a simple morphing effect by progressively warping between keyframes
        for i, img in enumerate(images):
            processed_images.append(img)
            
            # Add intermediate morphed frames between keyframes
            if i < num_images - 1:
                next_img = images[i+1]
                # Simple alpha blending as a basic "morphing" simulation
                for alpha in [0.25, 0.5, 0.75]:
                    morphed = Image.blend(img, next_img, alpha)
                    processed_images.append(morphed)
    
    elif effect == 'random':
        # Apply a random effect from the available ones
        effects = ['zoom-in', 'zoom-out', 'pan-left', 'pan-right', 'dissolve']
        random_effect = random.choice(effects)
        print(f"Applying random effect: {random_effect}")
        return apply_animation_effect(images, random_effect)
    
    else:
        print(f"Unknown effect '{effect}', returning original images")
        return images
    
    return processed_images if processed_images else images

def create_gif_from_images(images, output_path, duration=0.5):
    """Convert a list of PIL images to an animated GIF."""
    # Ensure all images are the same size (use the first image's size)
    base_width, base_height = images[0].size
    resized_images = []
    
    for img in images:
        if img.size != (base_width, base_height):
            img = img.resize((base_width, base_height), Image.Resampling.LANCZOS)
        # Convert to RGB if needed (removes alpha channel)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        resized_images.append(np.array(img))
    
    print(f"Creating GIF with {len(resized_images)} frames at {duration}s per frame")
    # Save as GIF
    imageio.mimsave(output_path, resized_images, format='GIF', duration=duration)
    return output_path

def create_mp4_from_images(images, output_path, fps=10, quality='high'):
    """Convert a list of PIL images to an MP4 video using FFmpeg."""
    # Create a temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save frames as PNG files
    print(f"Saving {len(images)} frames for video conversion...")
    for i, img in enumerate(images):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        img.save(frame_path)
    
    # Use FFmpeg to create video
    print(f"Creating MP4 video at {fps} FPS with {quality} quality...")
    try:
        # Set quality parameters based on quality setting
        if quality == 'ultra':
            crf = "15"  # Lower CRF = higher quality
            preset = "slow"
        elif quality == 'high':
            crf = "20"
            preset = "medium"
        else:  # standard
            crf = "23"
            preset = "faster"
            
        # Command to create video with FFmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',  # Use H.264 codec
            '-profile:v', 'high',
            '-preset', preset,  # Encoding speed/compression trade-off
            '-crf', crf,  # Quality (lower is better)
            '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
            output_path
        ]
        
        # Run FFmpeg
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'None'}")
        return None
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg to create videos.")
        return None
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return output_path

def generate_frames_with_latent_interpolation(pipe, prompt, num_frames=6, height=512, width=512, 
                                             guidance_scale=7.5, seed=None, num_inference_steps=50, 
                                             quality='high', interpolation_steps=8, animation_effect=None,
                                             transition_type='smooth'):
    """Generate frames using latent space interpolation for smoother transitions."""
    # Set quality parameters
    if quality == 'ultra':
        num_inference_steps = 100  # More steps for higher quality
        num_interpolation_steps = interpolation_steps if interpolation_steps else 12  # More interpolation steps for smoother animation
    elif quality == 'high':
        num_inference_steps = 50
        num_interpolation_steps = interpolation_steps if interpolation_steps else 8
    else:  # standard
        num_inference_steps = 30
        num_interpolation_steps = interpolation_steps if interpolation_steps else 4
    
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        generator = torch.Generator(device=pipe.device).manual_seed(random.randint(0, 2147483647))
    
    # Generate keyframes first
    print(f"Generating {num_frames} keyframes...")
    
    # Get latents for start and end points (and optionally middle points for longer animations)
    latents = []
    prompts = []
    
    # Parse style presets from prompt
    style_prompt = ""
    base_prompt = prompt
    
    # Look for style tags like [surreal], [fantasy], etc.
    style_match = re.search(r'\[(surreal|fantasy|horror|anime|abstract|cinematic|psychedelic)\]', prompt)
    if style_match:
        style = style_match.group(1)
        style_prompt = add_style_prompt(style)
        # Remove the style tag from the base prompt
        base_prompt = re.sub(r'\[(surreal|fantasy|horror|anime|abstract|cinematic|psychedelic)\]', '', prompt).strip()
    
    # Create slightly varied prompts for each keyframe
    for i in range(num_frames):
        # Add slight variation to prompts
        if i == 0:
            frame_prompt = f"{style_prompt} {base_prompt} Start scene."
        elif i == num_frames - 1:
            frame_prompt = f"{style_prompt} {base_prompt} End scene."
        else:
            frame_prompt = f"{style_prompt} {base_prompt} Scene {i+1}."
        
        prompts.append(frame_prompt)
    
    # Generate latents for each keyframe
    for i, frame_prompt in enumerate(prompts):
        print(f"Generating latent for keyframe {i+1}/{len(prompts)}...")
        # Get random latent noise
        latent = torch.randn(
            (1, pipe.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=pipe.device,
            dtype=torch.float16 if pipe.device.type == "cuda" else torch.float32
        )
        latents.append(latent)
    
    # Now generate all the frames with interpolated latents
    all_frames = []
    frame_count = 0
    
    # Process each pair of keyframes and interpolate between them
    for i in range(len(latents) - 1):
        # Get latents for current pair
        start_latent = latents[i]
        end_latent = latents[i + 1]
        
        # For each pair, generate interpolated frames
        steps = num_interpolation_steps
        for step in range(steps):
            # Calculate interpolation ratio
            alpha = step / steps
            
            # Apply different transition types
            if transition_type == 'smooth':
                # Default smooth interpolation
                interpolated_latent = (1 - alpha) * start_latent + alpha * end_latent
            elif transition_type == 'fade':
                # Fade transition - cubic easing
                cubic_alpha = alpha * alpha * (3 - 2 * alpha)
                interpolated_latent = (1 - cubic_alpha) * start_latent + cubic_alpha * end_latent
            elif transition_type == 'wipe':
                # Simplistic "wipe" effect - just use standard interpolation for now
                interpolated_latent = (1 - alpha) * start_latent + alpha * end_latent
            elif transition_type == 'dissolve':
                # Simplistic "dissolve" - steps with variable interpolation
                random_factor = 0.05 * random.random()
                adjusted_alpha = alpha + random_factor if alpha < 0.5 else alpha - random_factor
                adjusted_alpha = max(0, min(1, adjusted_alpha))  # Clamp to [0,1]
                interpolated_latent = (1 - adjusted_alpha) * start_latent + adjusted_alpha * end_latent
            else:
                # Default to smooth interpolation
                interpolated_latent = (1 - alpha) * start_latent + alpha * end_latent
            
            # Interpolate prompts too if you want
            if alpha < 0.5:
                current_prompt = prompts[i]
            else:
                current_prompt = prompts[i + 1]
            
            print(f"Generating frame {frame_count+1} (interpolation {step+1}/{steps} between keyframes {i+1} and {i+2})")
            
            # Generate image from interpolated latent
            with torch.no_grad():
                # Decode the latent to image
                image = pipe(
                    prompt=current_prompt,
                    latents=interpolated_latent,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]
            
            all_frames.append(image)
            frame_count += 1
    
    # Add the final keyframe
    with torch.no_grad():
        image = pipe(
            prompt=prompts[-1],
            latents=latents[-1],
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
    all_frames.append(image)
    
    # Apply animation effects if specified
    if animation_effect and animation_effect != 'none':
        print(f"Applying animation effect: {animation_effect}")
        all_frames = apply_animation_effect(all_frames, animation_effect)
    
    return all_frames

def add_style_prompt(style):
    """Add specific prompt additions based on the chosen style."""
    style_prompts = {
        'surreal': "dreamlike, surreal, Salvador Dali style, melting objects, impossible physics,",
        'fantasy': "fantasy art, magical, ethereal, mystical landscape, fantasy world, digital art,",
        'horror': "dark, eerie, horror, unsettling, creepy atmosphere, horror movie scene,",
        'anime': "anime style, vibrant, colorful, manga inspired, studio ghibli, detailed anime art,",
        'abstract': "abstract art, non-representational, geometric shapes, color fields, expressionist,",
        'cinematic': "cinematic, film still, dramatic lighting, movie scene, professional photography,",
        'psychedelic': "psychedelic art, vibrant colors, fractal patterns, kaleidoscopic, visionary art,"
    }
    
    return style_prompts.get(style, "")

def generate_narration(prompt, narration_type='story'):
    """Generate a narration for the dream video based on the prompt."""
    try:
        import json
        import subprocess
        
        print(f"Generating {narration_type} narration for the dream...")
        
        # Use the generate_dream_story.py script if available
        cmd = [
            'python3', 'generate_dream_story.py',
            '--prompt', prompt,
            '--type', narration_type
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the JSON output
        try:
            output = json.loads(result.stdout)
            if 'story' in output:
                return output['story']
            elif 'interpretation' in output:
                return output['interpretation']
            elif 'text' in output:
                return output['text']
            else:
                return output.get('content', "Failed to generate narration")
        except json.JSONDecodeError:
            # If not JSON, just return the raw output
            return result.stdout.strip()
    
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate narration: {e}")
        print(f"Error output: {e.stderr}")
        
        # Fallback method if the script is not available
        fallback_narrations = {
            'story': f"In my dream, {prompt}. As I wandered through this dreamscape, I felt a sense of wonder and curiosity. The images shifted and changed, revealing new aspects of my subconscious mind.",
            'interpretation': f"This dream about '{prompt}' suggests themes of exploration and discovery in your subconscious. The imagery reflects your inner thoughts and emotions, possibly revealing hidden aspects of yourself that you're beginning to recognize.",
            'poetic': f"Through misty veils of slumber,\nVisions of {prompt} emerge.\nDreams like whispers in the dark,\nRevealing truths we've yet to mark."
        }
        
        return fallback_narrations.get(narration_type, f"A dream about {prompt}.")
    
    except Exception as e:
        print(f"Unexpected error generating narration: {e}")
        return f"A dream about {prompt}."

def main():
    parser = argparse.ArgumentParser(description="Generate animated GIF or MP4 video from dream description.")
    parser.add_argument('--prompt', type=str, required=True, help='Dream description prompt')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the generated GIF/MP4')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of keyframes to generate')
    parser.add_argument('--fps', type=float, default=10.0, help='Frames per second for the animation')
    parser.add_argument('--model', type=str, default='stabilityai/stable-diffusion-2-1', help='Model to use')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--width', type=int, default=512, help='Width of the generated images')
    parser.add_argument('--height', type=int, default=512, help='Height of the generated images')
    parser.add_argument('--format', type=str, default='gif', choices=['gif', 'mp4', 'both'], 
                        help='Output format: gif, mp4, or both')
    parser.add_argument('--guidance_scale', type=float, default=7.5, 
                        help='Guidance scale for Stable Diffusion (higher values adhere more closely to the prompt)')
    parser.add_argument('--quality', type=str, default='high', choices=['standard', 'high', 'ultra'],
                        help='Quality level affecting number of steps and interpolation')
    parser.add_argument('--num_inference_steps', type=int, default=None,
                        help='Number of inference steps (overrides quality setting if provided)')
    parser.add_argument('--interpolation_steps', type=int, default=None,
                        help='Number of interpolation steps between keyframes')
    parser.add_argument('--animation_effect', type=str, default=None,
                        choices=['none', 'zoom-in', 'zoom-out', 'pan-left', 'pan-right', 'dissolve', 'morph', 'random'],
                        help='Animation effect to apply to the generated frames')
    parser.add_argument('--transition_type', type=str, default='smooth',
                        choices=['smooth', 'fade', 'wipe', 'dissolve'],
                        help='Type of transition between keyframes')
    parser.add_argument('--narration', action='store_true', help='Generate a narration for the video')
    parser.add_argument('--narration_type', type=str, default='story',
                        choices=['story', 'interpretation', 'poetic'],
                        help='Type of narration to generate')
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disable safety checker for creative freedom
    )
    pipe = pipe.to(device)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Use num_inference_steps from args if provided, otherwise it will be set by quality in the function
    num_inference_steps = args.num_inference_steps
    
    # Generate frames with latent interpolation (smoother transitions)
    frames = generate_frames_with_latent_interpolation(
        pipe=pipe,
        prompt=args.prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=num_inference_steps,
        quality=args.quality,
        interpolation_steps=args.interpolation_steps,
        animation_effect=args.animation_effect,
        transition_type=args.transition_type
    )
    
    # Save individual frames for reference
    frames_dir = os.path.join(os.path.dirname(args.output_path), "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frames_dir, f"frame_{i+1:03d}.png")
        frame.save(frame_path)
        print(f"Saved frame: {frame_path}")
    
    # Determine output format and create appropriate files
    base_path = os.path.splitext(args.output_path)[0]
    
    if args.format in ['gif', 'both']:
        # Create and save the GIF
        gif_path = f"{base_path}.gif"
        duration = 1.0 / args.fps  # Convert fps to duration
        create_gif_from_images(frames, gif_path, duration)
        print(f"GIF saved to: {gif_path}")
    
    if args.format in ['mp4', 'both']:
        # Create and save the MP4
        mp4_path = f"{base_path}.mp4"
        create_mp4_from_images(frames, mp4_path, args.fps, args.quality)
        print(f"MP4 saved to: {mp4_path}")
    
    # Generate narration if requested
    if args.narration:
        try:
            narration = generate_narration(args.prompt, args.narration_type)
            narration_path = f"{base_path}_narration.txt"
            with open(narration_path, 'w') as f:
                f.write(narration)
            print(f"Narration saved to: {narration_path}")
        except Exception as e:
            print(f"Failed to generate narration: {e}")

if __name__ == "__main__":
    main() 