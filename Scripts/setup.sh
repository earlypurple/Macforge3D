#!/bin/bash
#
# MacForge3D Universal Setup Script
#
# This script automates the setup of the development environment for the MacForge3D application.
# It is designed to be idempotent and work on macOS (Apple Silicon, Intel) and Linux.
#
# It performs the following steps:
# 1. Detects the OS and architecture to determine the correct Homebrew path.
# 2. Installs Homebrew if it is not already present.
# 3. Installs system dependencies (Python 3.11, Git LFS) via Homebrew.
# 4. Creates a Python virtual environment.
# 5. Installs required Python packages from a requirements file.
# 6. Configures Git LFS.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting MacForge3D universal project setup..."

# --- 1. Detect OS and Homebrew Prefix ---
OS="$(uname -s)"
ARCH="$(uname -m)"
BREW_PREFIX=""

if [ "$OS" == "Darwin" ]; then
    # macOS
    if [ "$ARCH" == "arm64" ]; then
        # Apple Silicon
        BREW_PREFIX="/opt/homebrew"
    else
        # Intel
        BREW_PREFIX="/usr/local"
    fi
    # On macOS, we should check for Xcode Command Line Tools
    if ! xcode-select -p &>/dev/null; then
        echo "› Xcode Command Line Tools not found. Please install them by running 'xcode-select --install' in your terminal and re-run this script."
        exit 1
    fi
    echo "✅ Xcode Command Line Tools found."

elif [ "$OS" == "Linux" ]; then
    # Linux
    BREW_PREFIX="/home/linuxbrew/.linuxbrew"
else
    echo "Unsupported OS: $OS"
    exit 1
fi
echo "✅ Detected OS: $OS ($ARCH). Homebrew prefix set to: $BREW_PREFIX"


# --- 2. Install Homebrew ---
if ! command -v brew &>/dev/null; then
    echo "› Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# --- 3. Set up Homebrew Shell Environment ---
eval "$($BREW_PREFIX/bin/brew shellenv)"

echo "› Homebrew is installed. Updating..."
brew update
echo "✅ Homebrew setup complete."


# --- 4. Install System Dependencies ---
echo "› Installing system dependencies (Python 3.11, Git LFS, libsndfile, SwiftLint, GCC)..."
# SwiftLint is only installed on macOS, but brew will handle this gracefully.
brew install python@3.11 git-lfs libsndfile swiftlint gcc
echo "✅ System dependencies installed."


# --- 5. Configure Python Virtual Environment ---
PYTHON_VERSION="3.11"
VENV_DIR="python_env"
PYTHON_EXECUTABLE="$BREW_PREFIX/opt/python@$PYTHON_VERSION/bin/python$PYTHON_VERSION"

if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    echo "🚨 Python executable not found at $PYTHON_EXECUTABLE"
    # Fallback for some systems
    PYTHON_EXECUTABLE=$(which python3.11)
    if [ -z "$PYTHON_EXECUTABLE" ]; then
        echo "🚨 Could not find python 3.11 executable. Please install it."
        exit 1
    fi
fi


if [ -d "$VENV_DIR" ]; then
  echo "› Python virtual environment '$VENV_DIR' already exists. Re-creating it..."
  rm -rf "$VENV_DIR"
fi

echo "› Creating Python $PYTHON_VERSION virtual environment in './$VENV_DIR'..."
"$PYTHON_EXECUTABLE" -m venv "$VENV_DIR"
echo "✅ Virtual environment created."


# --- 6. Install Python packages ---
echo "› Activating virtual environment and installing Python packages..."
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

# Install packages from requirements.txt
echo "› Installing packages from Python/requirements.txt..."
# Install a CPU-only version of torch to save space
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Install other packages, excluding torch
grep -v '^torch==' Python/requirements.txt | pip install -r /dev/stdin

# Install development packages from requirements-dev.txt
echo "› Installing development packages from Python/requirements-dev.txt..."
pip install -r Python/requirements-dev.txt

echo "✅ Python packages installed."
deactivate
echo "› Virtual environment deactivated."


# --- 7. Configure Git LFS ---
echo "› Setting up Git LFS..."
git lfs install
echo "✅ Git LFS configured."

echo "🎉 MacForge3D setup complete!"
echo "To activate the Python environment for development, run: source $VENV_DIR/bin/activate"
echo "You can now open the MacForge3D.xcodeproj file in Xcode and build the project (on macOS)."
