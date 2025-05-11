import argparse
import json
import requests
import sys

def generate_story(prompt, story_type="sequel"):
    """Generate a story based on the dream description using Ollama's API."""
    
    # Craft the prompt based on the story type
    if story_type == "prequel":
        system_prompt = "You are a creative storyteller. Write a prequel story (what happened before) based on this dream description. Make it engaging, detailed, and about 300-500 words."
    elif story_type == "sequel":
        system_prompt = "You are a creative storyteller. Write a sequel story (what happens next) based on this dream description. Make it engaging, detailed, and about 300-500 words."
    else:  # continuation
        system_prompt = "You are a creative storyteller. Continue this dream narrative as if it were a story. Make it engaging, detailed, and about 300-500 words."
    
    # Prepare the request to Ollama API
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.2",
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result["response"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}", file=sys.stderr)
        return f"Error generating story: {e}"

def main():
    parser = argparse.ArgumentParser(description="Generate a story based on a dream using Ollama.")
    parser.add_argument('--prompt', type=str, required=True, help='Dream description or interpretation')
    parser.add_argument('--type', type=str, default='sequel', choices=['sequel', 'prequel', 'continuation'], 
                        help='Type of story to generate (sequel, prequel, continuation)')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    args = parser.parse_args()
    
    # Generate the story
    story = generate_story(args.prompt, args.type)
    
    # Output the story
    if args.output:
        with open(args.output, 'w') as f:
            f.write(story)
    else:
        result = {"story": story, "type": args.type}
        print(json.dumps(result))

if __name__ == "__main__":
    main() 