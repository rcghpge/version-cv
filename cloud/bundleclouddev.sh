#!/bin/bash

# Set project directory name
PROJECT_DIR="<project directoy here>"
ARCHIVE_NAME="<directory archived here>.tar.gz"

# Ensure you are in the correct parent directory
if [ ! -d "$PROJECT_DIR" ]; then
  echo "❌ Error: '$PROJECT_DIR' directory not found."
  exit 1
fi

echo "🧹 Cleaning and archiving '$PROJECT_DIR' into '$ARCHIVE_NAME'..."

# Create a clean tar.gz archive, excluding clutter
tar --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.log' \
    --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints' \
    --exclude='.pixi' \
    --exclude='.DS_Store' \
    --exclude='*.swp' \
    --exclude='.env' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='.venv' \
    -czvf "$ARCHIVE_NAME" "$PROJECT_DIR"

echo "✅ Archive created: $ARCHIVE_NAME"
