import os
import sys
import pytest

# Add the project root to the Python path
# This is necessary for the test to find the `Python` module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from Python.ai_models.figurine_generator_light import generate_figurine

@pytest.fixture
def cleanup_generated_files():
    """A pytest fixture to clean up generated files after a test."""
    generated_files = []
    yield generated_files
    # Teardown: clean up the files
    for f in generated_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"🧹 Cleaned up {f}")

def test_generate_figurine_light():
    """
    Tests the light version of the figurine generator.
    It should create a new .ply file by copying a placeholder.
    """
    # --- Arrange ---
    test_prompt = "a_test_figurine"

    # --- Act ---
    output_path = generate_figurine(test_prompt, quality="test")

    # --- Assert ---
    assert output_path is not None, "The function should return a path."
    assert "Error" not in output_path, f"The function returned an error: {output_path}"
    assert os.path.exists(output_path), f"The output file was not created at {output_path}"
    assert test_prompt in output_path, "The output filename should contain the prompt."
    assert "light.ply" in output_path, "The output filename should indicate it's a light model."

    # --- Cleanup ---
    # Even though we have a fixture, we can also clean up immediately
    # to be explicit. The fixture is a safety net.
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"🧹 Cleaned up {output_path}")

def test_generate_figurine_petit(cleanup_generated_files):
    """
    Tests the 'petit' quality setting for the light generator.
    It should create a new .ply file and the filename should reflect the quality.
    """
    # --- Arrange ---
    test_prompt = "a_tiny_dragon"
    quality = "petit"

    # --- Act ---
    output_path = generate_figurine(test_prompt, quality=quality)
    cleanup_generated_files.append(output_path)

    # --- Assert ---
    assert output_path is not None
    assert "Error" not in output_path
    assert os.path.exists(output_path)
    assert test_prompt in output_path
    assert f"_{quality}_light.ply" in output_path

def test_generate_figurine_no_placeholder():
    """
    Tests that the generator handles a missing placeholder file gracefully.
    """
    # --- Arrange ---
    placeholder_dir = "Examples/generated_figurines"
    backup_dir = "Examples/generated_figurines_bak"
    os.makedirs(backup_dir, exist_ok=True)

    # Move all placeholder files to the backup directory
    placeholders = [f for f in os.listdir(placeholder_dir) if f.endswith(".ply")]
    for p in placeholders:
        os.rename(os.path.join(placeholder_dir, p), os.path.join(backup_dir, p))

    # --- Act ---
    output_path = generate_figurine("any_prompt")

    # --- Assert ---
    assert "No placeholder models found" in output_path

    # --- Teardown ---
    # Restore the placeholders
    for p in placeholders:
        os.rename(os.path.join(backup_dir, p), os.path.join(placeholder_dir, p))
    os.rmdir(backup_dir)
