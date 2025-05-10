#!/bin/bash

# Initialize Git LFS if not already initialized
git lfs install

# Track all files except target directory
git lfs track "*" "!target/**"

# Remove target directory from git tracking
git rm -r --cached target/

# Add all files except those in .gitignore
git add .

# Commit changes
git commit -m "Configure Git LFS tracking and remove target directory"

# Push to GitHub
git push

echo "Changes pushed successfully!" 