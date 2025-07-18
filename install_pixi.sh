#!/usr/bin/env bash
set -e

# Install Pixi if not already installed
if ! command -v pixi &> /dev/null
then
    echo "Pixi not found, installing..."
    curl -sSf https://pixi.sh/install.sh | bash
else
    echo "Pixi is already installed."
fi

# Initialize Pixi (creates pixi.toml and pixi.lock if not present)
if [ ! -f "pixi.toml" ]; then
    echo "Initializing Pixi project..."
    pixi init
else
    echo "pixi.toml already exists, skipping init."
fi

# Install dependencies from pixi.toml
echo "Installing dependencies..."
pixi install

# Enter Pixi shell
echo "Entering Pixi environment..."
pixi shell
