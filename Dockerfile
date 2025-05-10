FROM rust:slim-bullseye as builder

# Install only essential build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a new empty shell project
WORKDIR /app
COPY . .

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bullseye-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary and necessary files
COPY --from=builder /app/target/release/dream_visualizer /app/dream_visualizer
COPY ./templates /app/templates
COPY ./static /app/static
COPY ./.env.example /app/.env.example

# Create data directories
RUN mkdir -p /app/data/images /app/data/videos

# Set environment variables
ENV SERVER_HOST=0.0.0.0

# Start the application
CMD ["./dream_visualizer"]

# Expose the web server port
EXPOSE 3000 