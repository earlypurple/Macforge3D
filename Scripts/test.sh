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

# Activate python venv to run checks
source "$VENV_DIR/bin/activate"

echo "› Checking Python code formatting with Black..."
black --check .
echo "✅ Black formatting check passed."

echo "› Checking Python type hints with mypy..."
# We need to target specific directories that have source code
mypy Python/ai_models Python/exporters Python/simulation
echo "✅ Mypy type checking passed."

# Add the Python directory to the python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/Python

# Run pytest using the virtual environment's python and check coverage
echo "› Running python tests with coverage..."
pytest \
  --cov=Python/ai_models \
  --cov=Python/exporters \
  --cov=Python/simulation \
  --cov-report=term-missing \
  --cov-fail-under=80 \
  Tests/python/

echo "✅ Python tests and coverage check passed."

# Deactivate venv
deactivate


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

# Check for SwiftLint
if ! command -v swiftlint &> /dev/null; then
    echo "⚠️ SwiftLint not found. Skipping Swift linting. Please install it by running 'brew install swiftlint'."
else
    echo "› Linting Swift code with SwiftLint..."
    swiftlint
    echo "✅ SwiftLint check passed."
fi

# Define project and scheme
PROJECT="MacForge3D.xcodeproj"
SCHEME="MacForge3D"

# Detect architecture for xcodebuild
ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "arm64" ]; then
    echo "🚨 Unsupported architecture: $ARCH. Only x86_64 and arm64 are supported for macOS tests."
    exit 1
fi
echo "› Detected architecture: $ARCH"

# Run the tests
xcodebuild test \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -destination "platform=macOS,arch=$ARCH" \
  -enableCodeCoverage YES

echo "✅ Swift/macOS tests passed."


echo "🎉 All tests passed successfully!"
