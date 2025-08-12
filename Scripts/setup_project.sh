#!/bin/bash
#
# MacForge3D Project Setup Script
#
# This script automates the setup of the development environment for the MacForge3D application.
# It installs system dependencies using Homebrew, sets up a Python virtual environment,
# and installs the required Python packages.
#
# Prerequisites:
# - macOS
# - Xcode Command Line Tools (can be installed with `xcode-select --install`)

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting MacForge3D project setup..."

# --- Check for and install Xcode Command Line Tools ---
if ! xcode-select -p &>/dev/null; then
  echo "› Xcode Command Line Tools not found. Please install them by running 'xcode-select --install' in your terminal and re-run this script."
  exit 1
fi
echo "✅ Xcode Command Line Tools found."

# --- Check for and install Homebrew ---
if ! command -v brew &>/dev/null; then
  echo "› Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add Homebrew to PATH for this script's session
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  echo "› Homebrew is already installed. Updating..."
  brew update
fi
echo "✅ Homebrew setup complete."

# --- Install system dependencies ---
echo "› Installing system dependencies (Python 3.11, Git LFS)..."
brew install python@3.11 git-lfs
echo "✅ System dependencies installed."

# --- Configure Python Virtual Environment ---
PYTHON_VERSION="3.11"
VENV_DIR="python_env"

if [ -d "$VENV_DIR" ]; then
  echo "› Python virtual environment '$VENV_DIR' already exists. Re-creating it..."
  rm -rf "$VENV_DIR"
fi

echo "› Creating Python $PYTHON_VERSION virtual environment in './$VENV_DIR'..."
# On Apple Silicon, Homebrew installs to /opt/homebrew. On Intel, /usr/local.
# This logic handles both cases.
if [ -f "/opt/homebrew/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION" ]; then
    "/opt/homebrew/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION" -m venv "$VENV_DIR"
else
    "/usr/local/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION" -m venv "$VENV_DIR"
fi
echo "✅ Virtual environment created."

# --- Activate virtual environment and install Python packages ---
echo "› Activating virtual environment and installing Python packages..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install required packages for AI and 3D processing
# torch, transformers, diffusers for AI models
# Pillow for image handling
# numpy for numerical operations
# trimesh for 3D mesh processing (useful for later)
pip install torch transformers diffusers accelerate Pillow numpy trimesh safetensors
# Note: PythonKit is a Swift framework, not a pip package. It's already in the project.

echo "✅ Python packages installed."
deactivate
echo "› Virtual environment deactivated."


# --- Configure Git LFS ---
echo "› Setting up Git LFS..."
git lfs install
# The following command would be needed to track large files,
# but we will let the developer decide which files to track.
# Example: git lfs track "*.ply"
# We will just ensure it's installed for the repo.
echo "✅ Git LFS configured."

echo "🎉 MacForge3D setup complete!"
echo "To activate the Python environment for development, run: source $VENV_DIR/bin/activate"
echo "You can now open the MacForge3D.xcodeproj file in Xcode and build the project."
