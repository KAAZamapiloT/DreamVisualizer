#!/bin/bash

# Dream Visualizer starter script
# This script helps launch the Dream Visualizer application with different configurations

# Default values
MODE="web"
ENV_FILE=".env"
DREAM_TEXT=""
BUILD_MODE="debug"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo -e "${BLUE}Dream Visualizer${NC} - AI-powered dream visualization"
    echo
    echo "Usage: ./start.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  -m, --mode [web|cli]        Run in web or cli mode (default: web)"
    echo "  -e, --env [filename]        Use specific .env file (default: .env)"
    echo "  -d, --dream \"text\"          Dream text to process (CLI mode only)"
    echo "  -o, --open-source           Use open source configuration"
    echo "  -c, --commercial            Use commercial API configuration"
    echo "  -r, --release               Run in release mode"
    echo "  -h, --help                  Show this help message"
    echo
    echo "Examples:"
    echo "  ./start.sh                  Start web server with default configuration"
    echo "  ./start.sh --mode cli --dream \"I was flying\" --open-source"
    echo "                              Process dream with open source models"
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -e|--env)
            ENV_FILE="$2"
            shift 2
            ;;
        -d|--dream)
            DREAM_TEXT="$2"
            shift 2
            ;;
        -o|--open-source)
            # Set open source configuration
            export LOCAL_LLM_ENABLED=true
            export SD_USE_LOCAL_API=true
            echo -e "${GREEN}Using open source models${NC}"
            shift
            ;;
        -c|--commercial)
            # Set commercial API configuration
            export LOCAL_LLM_ENABLED=false
            export SD_USE_LOCAL_API=false
            echo -e "${YELLOW}Using commercial APIs${NC}"
            shift
            ;;
        -r|--release)
            BUILD_MODE="release"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            show_help
            exit 1
            ;;
    esac
done

# Check if required files exist
if [ ! -f "$ENV_FILE" ] && [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: No .env file found. Creating from .env.example${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}Created .env from example. Please edit as needed.${NC}"
    else
        echo -e "${RED}Error: No .env.example file found.${NC}"
        exit 1
    fi
fi

# Load environment variables if file exists
if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}Loading configuration from $ENV_FILE${NC}"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Check if we need web or CLI mode
if [ "$MODE" = "web" ]; then
    echo -e "${GREEN}Starting Dream Visualizer in Web mode${NC}"
    echo -e "Open ${BLUE}http://${SERVER_HOST:-127.0.0.1}:${SERVER_PORT:-3000}${NC} in your browser"
    if [ "$BUILD_MODE" = "release" ]; then
        cargo run --release
    else
        cargo run
    fi
elif [ "$MODE" = "cli" ]; then
    if [ -z "$DREAM_TEXT" ]; then
        echo -e "${YELLOW}No dream text provided. Will prompt for input.${NC}"
        if [ "$BUILD_MODE" = "release" ]; then
            cargo run --release -- --cli
        else
            cargo run -- --cli
        fi
    else
        echo -e "${GREEN}Processing dream: ${BLUE}$DREAM_TEXT${NC}"
        if [ "$BUILD_MODE" = "release" ]; then
            cargo run --release -- --cli --dream "$DREAM_TEXT"
        else
            cargo run -- --cli --dream "$DREAM_TEXT"
        fi
    fi
else
    echo -e "${RED}Invalid mode: $MODE${NC}"
    show_help
    exit 1
fi 