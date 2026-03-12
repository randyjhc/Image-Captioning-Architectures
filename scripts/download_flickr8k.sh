#!/bin/bash

# Script to download Flickr8k dataset from Kaggle
# This dataset contains 8,000 images with 5 captions each

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATASET_NAME="adityajn105/flickr8k"
TARGET_DIR="data/datasets/flickr8k"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}Flickr8k Dataset Download Script${NC}"
echo "=================================="
echo ""

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo -e "${RED}Error: Kaggle CLI is not installed${NC}"
    echo "Please install it using: pip install kaggle"
    echo "Or with uv: uv pip install kaggle"
    exit 1
fi

echo -e "${GREEN}✓${NC} Kaggle CLI found"

# Check for Kaggle API credentials
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"

if [ ! -f "$KAGGLE_JSON" ]; then
    echo -e "${RED}Error: Kaggle API credentials not found${NC}"
    echo ""
    echo "To set up Kaggle API credentials:"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Scroll to 'API' section and click 'Create Legacy API Key'"
    echo "  3. This will automatically download a kaggle.json file to your Downloads folder"
    echo "  4. Move the downloaded file to the correct location:"
    echo "     mkdir -p ~/.kaggle"
    echo "     mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "     chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo -e "${GREEN}✓${NC} Kaggle credentials found at $KAGGLE_JSON"

# Verify proper file permissions for security
PERMS=$(stat -c "%a" "$KAGGLE_JSON" 2>/dev/null || stat -f "%Lp" "$KAGGLE_JSON" 2>/dev/null)
if [ "$PERMS" != "600" ]; then
    echo -e "${YELLOW}Warning: kaggle.json has insecure permissions ($PERMS)${NC}"
    echo "Fixing permissions..."
    chmod 600 "$KAGGLE_JSON"
    echo -e "${GREEN}✓${NC} Permissions set to 600"
fi

# Create target directory
cd "$PROJECT_ROOT"
mkdir -p "$TARGET_DIR"

# Check if dataset already exists
if [ -d "$TARGET_DIR" ] && [ "$(ls -A $TARGET_DIR)" ]; then
    echo ""
    echo -e "${YELLOW}Warning: Dataset directory is not empty${NC}"
    echo "Found existing files in: $TARGET_DIR"
    echo ""
    ls -lh "$TARGET_DIR"
    echo ""
    read -p "Do you want to re-download and overwrite? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Skipping download. Using existing dataset.${NC}"
        exit 0
    fi
    echo "Removing existing files..."
    rm -rf "$TARGET_DIR"/*
fi

echo ""
echo "Downloading Flickr8k dataset from Kaggle..."
echo "Dataset: $DATASET_NAME"
echo "Target directory: $TARGET_DIR"
echo ""

# Download the dataset
kaggle datasets download -d "$DATASET_NAME" -p "$TARGET_DIR" --unzip

echo ""
echo -e "${GREEN}✓ Download completed!${NC}"
echo ""

# Show the downloaded structure
echo "Downloaded files:"
ls -lh "$TARGET_DIR"

echo ""
echo -e "${GREEN}Dataset successfully downloaded to: $TARGET_DIR${NC}"
echo ""
