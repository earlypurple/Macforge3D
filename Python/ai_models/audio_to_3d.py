import os
import numpy as np
import librosa
import trimesh
from datetime import datetime

def generate_3d_from_audio(audio_path: str, output_dir: str = "Examples/generated_audio_models") -> str:
    """
    Generates a 3D model from an audio file using a procedural approach.

    The function analyzes audio features and uses them to deform a base mesh,
    creating a unique 3D shape that represents the audio's characteristics.

    Args:
        audio_path: The path to the input audio file.
        output_dir: The directory where the generated .ply file will be saved.

    Returns:
        The path to the generated .ply file, or an error message string.
    """
    print(f"🐍 [Audio-to-3D] Starting generation from: '{audio_path}'")

    # --- 1. Load Audio and Extract Features ---
    try:
        import soundfile as sf
        # Load the audio file using soundfile to avoid librosa's backend issues.
        y, sr = sf.read(audio_path)

        # Ensure the audio is mono for librosa's feature extraction
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample to a standard rate if necessary
        target_sr = 22050
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Limit to 30 seconds
        max_duration = 30
        if len(y) > sr * max_duration:
            y = y[:sr * max_duration]

        # Extract Root Mean Square (RMS) to represent overall loudness/energy.
        # We'll use the average energy.
        rms = np.mean(librosa.feature.rms(y=y))

        # Extract a Chromagram, which represents the 12 pitch classes (C, C#, D, etc.).
        # This gives us a harmonic profile of the audio.
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average the chroma features over time to get a single vector of 12 values.
        chroma_features = np.mean(chroma, axis=1)

        print(f"✅ [Audio-to-3D] Audio features extracted successfully.")
        print(f"    - Average RMS (Energy): {rms:.4f}")
        print(f"    - Chroma Features: {np.round(chroma_features, 2)}")

    except Exception as e:
        error_msg = f"Error loading or analyzing audio file: {e}"
        print(f"❌ [Audio-to-3D] {error_msg}")
        return error_msg

    # --- 2. Generate Base 3D Mesh ---
    try:
        # Create a base icosphere. It's a sphere made of triangles, good for deformation.
        # A subdivision of 4 gives a good balance of detail and performance.
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        print(f"✅ [Audio-to-3D] Base icosphere created with {len(mesh.vertices)} vertices.")
    except Exception as e:
        error_msg = f"Error creating base mesh: {e}"
        print(f"❌ [Audio-to-3D] {error_msg}")
        return error_msg

    # --- 3. Deform Mesh Based on Audio Features ---
    print("🐍 [Audio-to-3D] Deforming mesh based on audio features...")
    # Get the original vertex coordinates.
    vertices = mesh.vertices.copy()

    # Normalize the chroma features to be between 0 and 1 for stable deformation.
    chroma_normalized = (chroma_features - chroma_features.min()) / (chroma_features.max() - chroma_features.min() + 1e-6)

    # Define 12 directions in 3D space, one for each pitch class.
    # These directions are spread out to create interesting deformations.
    directions = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        [0.7, 0.7, 0], [-0.7, 0.7, 0], [0.7, -0.7, 0], [-0.7, -0.7, 0],
        [0, 0.7, 0.7], [0, -0.7, 0.7]
    ])

    # Calculate the displacement for each vertex.
    # The core of the procedural generation.
    for i, vertex in enumerate(vertices):
        displacement = np.zeros(3)
        # For each of the 12 chroma features, push the vertex along the corresponding direction.
        for j in range(12):
            # The displacement is proportional to the energy of that pitch class.
            displacement += directions[j] * chroma_normalized[j]

        # The overall scale of the deformation is controlled by the audio's RMS (loudness).
        # A small constant factor is added to ensure even quiet sounds have some effect.
        scale_factor = (rms * 5.0) + 0.1

        # Apply the scaled displacement to the vertex's normal direction.
        # This makes the shape expand outwards from its center.
        normal = mesh.vertex_normals[i]
        vertices[i] += normal * np.linalg.norm(displacement) * scale_factor

    # Update the mesh with the new, deformed vertices.
    mesh.vertices = vertices
    print("✅ [Audio-to-3D] Mesh deformation complete.")

    # --- 4. Save the Resulting Mesh ---
    try:
        # Create the output directory if it doesn't exist.
        os.makedirs(output_dir, exist_ok=True)

        # Generate a unique filename.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = os.path.splitext(os.path.basename(audio_path))[0][:20]
        filename = f"{timestamp}_{audio_name}_audio3D.ply"
        output_path = os.path.join(output_dir, filename)

        # Export the mesh to a .ply file.
        mesh.export(output_path)

        print(f"✅ [Audio-to-3D] Model saved successfully to: {output_path}")
        return output_path
    except Exception as e:
        error_msg = f"Error saving the generated mesh: {e}"
        print(f"❌ [Audio-to-3D] {error_msg}")
        return error_msg

if __name__ == '__main__':
    # This block allows for testing the script directly from the command line.
    print("\n--- Running standalone test of audio_to_3d.py ---")

    # We need a sample audio file to test. Let's create a dummy one if it doesn't exist.
    # In a real scenario, the user provides this.
    test_audio_dir = "Tests/python/test_data"
    os.makedirs(test_audio_dir, exist_ok=True)
    test_audio_path = os.path.join(test_audio_dir, "sample_audio.wav")

    try:
        import soundfile as sf
        # Create a simple sine wave at 440 Hz (A4) for testing.
        sr_test = 22050
        duration_test = 5
        frequency = 440
        t = np.linspace(0., duration_test, int(sr_test * duration_test))
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        sf.write(test_audio_path, data.astype(np.int16), sr_test)
        print(f"✅ Created a dummy audio file for testing at: {test_audio_path}")
    except ImportError:
        print("⚠️ Could not create a dummy audio file because 'soundfile' is not installed.")
        print("   Please install it (`pip install soundfile`) or provide your own audio file for testing.")
        test_audio_path = None # Set to None to skip the test if we can't create the file.
    except Exception as e:
        print(f"⚠️ An error occurred while creating the dummy audio file: {e}")
        test_audio_path = None

    if test_audio_path and os.path.exists(test_audio_path):
        test_output_dir = "Examples/generated_audio_models_test"
        print(f"Test audio file: '{test_audio_path}'")
        print(f"Test output directory: '{test_output_dir}'")

        # Run the generation function
        path = generate_3d_from_audio(test_audio_path, output_dir=test_output_dir)

        if path and "Error" not in path:
            print(f"\n✅ Test successful! Model saved at: {path}")
            print(f"   You can view the generated .ply file in the '{test_output_dir}' directory.")
        else:
            print(f"\n❌ Test failed. Reason: {path}")
    else:
        print("\n--- Test skipped: No sample audio file available. ---")
