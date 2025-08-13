#!/bin/bash
#
# MacForge3D Universal Test Script
#
# This script runs all tests for the MacForge3D project, including:
# 1. Python unit tests using pytest.
# 2. Swift/macOS UI tests using xcodebuild.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting MacForge3D test suite..."

# --- 1. Run Python Tests ---
echo "🐍 Running Python tests..."

VENV_DIR="python_env"
PYTHON_EXEC="$VENV_DIR/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "🚨 Python executable not found in virtual environment. Please run Scripts/setup.sh first."
    exit 1
fi

# Run pytest using the virtual environment's python
"$PYTHON_EXEC" -m pytest Tests/python/

echo "✅ Python tests passed."


# --- 2. Run Swift/macOS UI Tests ---
echo "🍎 Running Swift/macOS UI tests..."

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "⚠️ Skipping Swift tests: Not running on macOS."
    exit 0
fi

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    echo "🚨 Xcode command line tools are not installed. Please install them and try again."
    exit 1
fi

# Define project and scheme
PROJECT="MacForge3D.xcodeproj"
SCHEME="MacForge3D"

# Run the tests
xcodebuild test \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -destination 'platform=macOS,arch=x86_64' \
  -enableCodeCoverage YES

echo "✅ Swift/macOS tests passed."


echo "🎉 All tests passed successfully!"
