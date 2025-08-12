#!/bin/bash
set -e
echo "🚀 Starting MacForge3D project setup (no Xcode check)..."
if ! command -v brew &>/dev/null; then
  echo "› Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  echo "› Homebrew is already installed. Updating..."
  brew update
fi
echo "✅ Homebrew setup complete."
echo "› Installing system dependencies (Python 3.11, Git LFS)..."
brew install python@3.11 git-lfs
echo "✅ System dependencies installed."
PYTHON_VERSION="3.11"
VENV_DIR="python_env"
if [ -d "$VENV_DIR" ]; then
  echo "› Python virtual environment '$VENV_DIR' already exists. Re-creating it..."
  rm -rf "$VENV_DIR"
fi
echo "› Creating Python $PYTHON_VERSION virtual environment in './$VENV_DIR'..."
if [ -f "/opt/homebrew/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION" ]; then
    "/opt/homebrew/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION" -m venv "$VENV_DIR"
else
    "/usr/local/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION" -m venv "$VENV_DIR"
fi
echo "✅ Virtual environment created."
echo "› Activating virtual environment and installing Python packages..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install torch transformers diffusers accelerate Pillow numpy trimesh safetensors
echo "✅ Python packages installed."
deactivate
echo "› Virtual environment deactivated."
echo "› Setting up Git LFS..."
git lfs install
echo "✅ Git LFS configured."
echo "🎉 MacForge3D setup complete!"
echo "To activate the Python environment for development, run: source $VENV_DIR/bin/activate"
echo "You can now open the MacForge3D.xcodeproj file in Xcode and build the project."
