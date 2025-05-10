# Dream Visualizer

A Rust application that transforms free-form dream descriptions into visual animations using AI, with support for both commercial APIs and open source models.

## Features

- **Text Analysis**: Extract key visual elements from dream descriptions using OpenAI or local LLMs
- **Image Generation**: Generate images using Stability AI or local Stable Diffusion WebUI
- **Animation Creation**: Stitch images into GIF or WebP animations using pure Rust libraries
- **Web Interface**: Modern web UI built with Axum and Tera templates
- **CLI Support**: Command-line interface for batch processing

## Open Source Model Support

This application supports using fully open source alternatives:

1. **Local Language Models**: Use models like LLaMA, Mistral, or other open source LLMs through an OpenAI-compatible API (such as [llama.cpp](https://github.com/ggerganov/llama.cpp) or [Ollama](https://ollama.ai/))

2. **Local Stable Diffusion**: Connect to [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) running locally for image generation

3. **Fallback to commercial APIs**: Use OpenAI and Stability APIs when local models aren't available

## Prerequisites

1. **Rust and Cargo**: Install from [rustup.rs](https://rustup.rs/)
2. **For local model support**:
   - A running Stable Diffusion WebUI with API access enabled
   - A running LLM server with OpenAI API compatibility

## Configuration

Configuration is handled through environment variables or a `.env` file. See `.env.example` for all options.

### Using Open Source Models

To use local models:

1. Set `LOCAL_LLM_ENABLED=true` and configure `LOCAL_LLM_API_ENDPOINT`
2. Set `SD_USE_LOCAL_API=true` and configure `SD_LOCAL_API_ENDPOINT`

Example .env for open source setup:
```
LOCAL_LLM_ENABLED=true
LOCAL_LLM_API_ENDPOINT=http://localhost:8080/v1
LOCAL_LLM_MODEL=llama3

SD_USE_LOCAL_API=true
SD_LOCAL_API_ENDPOINT=http://localhost:7860
```

## Installation

1. Clone the repository
2. Copy `.env.example` to `.env` and edit the settings
3. Build: `cargo build --release`

## Running the Application

### Using the Starter Script

The easiest way to start the application is with the included starter script:

```bash
# Start the web interface with default settings
./start.sh

# Use open source models instead of commercial APIs
./start.sh --open-source

# Process a dream in CLI mode
./start.sh --mode cli --dream "I dreamt of flying whales in a purple sky"

# Show all options
./start.sh --help
```

### Manual Startup

#### Web Mode (default)
```
cargo run --release
```

#### CLI Mode
```
cargo run --release -- --cli --dream "I dreamt I was flying over mountains made of crystal."
```

### Docker Deployment

The application can also be run using Docker:

```bash
# Build and run with Docker
docker build -t dream-visualizer .
docker run -p 3000:3000 dream-visualizer

# Or use docker-compose
docker-compose up dream-visualizer
```

To use with local open source models, edit the docker-compose.yml file to enable the Ollama and Stable Diffusion WebUI services.

## License

MIT
