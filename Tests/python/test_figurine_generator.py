import os
import sys
import pytest
from PIL import Image
import numpy as np

# Add the parent directory to the path so we can import the ai_models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Python.ai_models.figurine_generator import generate_figurine, initialize_pipelines


@pytest.fixture(scope="module")
def setup_pipelines():
    """Initializes pipelines once for the entire test module."""
    initialize_pipelines()


def test_generate_figurine_standard_quality(setup_pipelines):
    """Tests the standard quality text-to-3D generation."""
    prompt = "a cute cat"
    output_dir = "/tmp/test_figurine_generator"
    result = generate_figurine(prompt, quality="standard", output_dir=output_dir)

    assert "Error" not in result
    assert os.path.exists(result)
    assert result.endswith(".ply")


def test_generate_figurine_ultra_realistic(setup_pipelines):
    """Tests the ultra-realistic image-to-3D generation."""
    # Create a dummy image for testing
    dummy_image_path = "/tmp/test_image.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(dummy_image_path)

    output_dir = "/tmp/test_figurine_generator"
    result = generate_figurine(
        prompt="dummy prompt",
        quality="ultra_realistic",
        output_dir=output_dir,
        image_path=dummy_image_path,
    )

    assert "Error" not in result
    assert os.path.exists(result)
    assert result.endswith(".ply")


def test_generate_figurine_invalid_quality(setup_pipelines):
    """Tests that an invalid quality setting returns an error or handles it gracefully."""
    prompt = "a test"
    output_dir = "/tmp/test_figurine_generator"
    # This test assumes that the function might not raise an error but could return a path
    # to a default-quality model or handle it internally. If it should raise an error,
    # the test should be adjusted to use pytest.raises.
    result = generate_figurine(prompt, quality="invalid_quality", output_dir=output_dir)
    assert "Error" not in result  # or check for a specific error message if applicable


def test_generate_figurine_no_image_for_ultra_realistic(setup_pipelines):
    """Tests that an error is returned when no image is provided for ultra-realistic mode."""
    prompt = "a test"
    output_dir = "/tmp/test_figurine_generator"
    result = generate_figurine(
        prompt, quality="ultra_realistic", output_dir=output_dir, image_path=None
    )
    assert "Error: Image path is required" in result
