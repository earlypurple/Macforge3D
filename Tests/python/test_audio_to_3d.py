import os
import sys
import pytest
import numpy as np
import soundfile as sf
import trimesh

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from Python.ai_models.audio_to_3d import generate_3d_from_audio


@pytest.fixture
def audio_test_data():
    """
    A pytest fixture to create a dummy audio file for testing and clean it up afterward.
    It also cleans up the generated 3D model.
    """
    # --- Setup: Create a dummy audio file ---
    test_data_dir = os.path.join(project_root, "Tests/python/test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    test_audio_path = os.path.join(test_data_dir, "test_tone.wav")

    sr = 22050
    duration = 1
    frequency = 440
    t = np.linspace(0.0, duration, int(sr * duration))
    amplitude = np.iinfo(np.int16).max * 0.3
    data = amplitude * np.sin(2.0 * np.pi * frequency * t)
    sf.write(test_audio_path, data.astype(np.int16), sr)

    generated_files = [test_audio_path]

    # --- Yield the path to the test audio file ---
    # The 'yield' keyword passes the data to the test function.
    yield test_audio_path, generated_files

    # --- Teardown: Clean up all generated files ---
    print("\n🧹 Cleaning up generated test files...")
    for f_path in generated_files:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                print(f"   - Removed: {f_path}")
            except OSError as e:
                print(f"   - Error removing file {f_path}: {e}")


def test_generate_3d_from_audio_success(audio_test_data):
    """
    Tests the successful generation of a 3D model from an audio file.
    """
    # --- Arrange ---
    test_audio_path, generated_files = audio_test_data
    test_output_dir = os.path.join(project_root, "Tests/python/test_output")

    # --- Act ---
    output_path = generate_3d_from_audio(test_audio_path, output_dir=test_output_dir)
    generated_files.append(output_path)  # Add generated model to cleanup list

    # --- Assert ---
    assert output_path is not None, "Function should return a path string."
    assert "Error" not in output_path, f"Generation failed with an error: {output_path}"
    assert os.path.exists(
        output_path
    ), f"Output file was not created at the expected path: {output_path}"

    # Check that the file is a valid mesh and not empty
    try:
        mesh = trimesh.load(output_path)
        assert isinstance(
            mesh, trimesh.Trimesh
        ), "The output file is not a valid trimesh object."
        assert len(mesh.vertices) > 0, "The generated mesh has no vertices."
        assert len(mesh.faces) > 0, "The generated mesh has no faces."
    except Exception as e:
        pytest.fail(f"Failed to load or validate the generated mesh file: {e}")

    # Check filename format
    assert "test_tone" in output_path, "Filename should contain the audio file name."
    assert "audio3D.ply" in output_path, "Filename should end with the correct suffix."

    # --- Cleanup is handled by the fixture ---


def test_generate_3d_from_audio_file_not_found():
    """
    Tests that the function handles a non-existent audio file gracefully.
    """
    # --- Arrange ---
    non_existent_file = "path/to/non_existent_audio_file.wav"

    # --- Act ---
    output_path = generate_3d_from_audio(non_existent_file)

    # --- Assert ---
    assert "Error" in output_path, "Function should return an error for a missing file."
    assert (
        "System error" in output_path or "not found" in output_path
    ), "The error message should indicate that the file was not found."


def test_generate_3d_from_unsupported_file_type(tmp_path):
    """
    Tests that the function handles an unsupported file type gracefully.
    Librosa should raise an error, which the function should catch and return.
    """
    # --- Arrange ---
    # Create a dummy text file instead of an audio file
    unsupported_file = tmp_path / "not_an_audio_file.txt"
    unsupported_file.write_text("This is not audio data.")

    # --- Act ---
    output_path = generate_3d_from_audio(str(unsupported_file))

    # --- Assert ---
    assert (
        "Error" in output_path
    ), "Function should return an error for an unsupported file."
    assert (
        "Error loading or analyzing audio file" in output_path
    ), "The error message should be about failing to load/analyze the audio."
