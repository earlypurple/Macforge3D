import torch
from diffusers import ShapEPipeline, TripoSRPipeline
from diffusers.utils import export_to_ply
import trimesh
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
shap_e_model_name = "openai/shap-e"
tripo_sr_model_name = "stabilityai/TripoSR"

# --- Global Variables ---
pipe_shap_e = None
pipe_tripo_sr = None
device = None
torch_dtype = None

def initialize_pipelines():
    """Initializes all 3D generation pipelines and moves them to the appropriate device."""
    global pipe_shap_e, pipe_tripo_sr, device, torch_dtype, shap_e_model_name, tripo_sr_model_name

    if pipe_shap_e and pipe_tripo_sr:
        return

    # --- Setup device ---
    if not device:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            torch_dtype = torch.float16
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            torch_dtype = torch.float16
        else:
            device = torch.device("cpu")
            torch_dtype = torch.float32
        print(f"🐍 [Figurine Generator] Using device: {device} with dtype: {torch_dtype}")

    # --- Load Shap-E Pipeline ---
    if not pipe_shap_e:
        print(f"🐍 [Figurine Generator] Loading Shap-E pipeline from '{shap_e_model_name}'...")
        try:
            pipe_shap_e = ShapEPipeline.from_pretrained(shap_e_model_name, torch_dtype=torch_dtype)
            pipe_shap_e = pipe_shap_e.to(device)
            print("✅ [Figurine Generator] Shap-E pipeline loaded successfully.")
        except Exception as e:
            print(f"❌ [Figurine Generator] Failed to load the Shap-E pipeline: {e}")
            pipe_shap_e = None

    # --- Load TripoSR Pipeline ---
    if not pipe_tripo_sr:
        print(f"🐍 [Figurine Generator] Loading TripoSR pipeline from '{tripo_sr_model_name}'...")
        try:
            pipe_tripo_sr = TripoSRPipeline.from_pretrained(tripo_sr_model_name, torch_dtype=torch_dtype)
            pipe_tripo_sr = pipe_tripo_sr.to(device)
            print("✅ [Figurine Generator] TripoSR pipeline loaded successfully.")
        except Exception as e:
            print(f"❌ [Figurine Generator] Failed to load the TripoSR pipeline: {e}")
            pipe_tripo_sr = None

def _refine_mesh(ply_path: str, iterations: int = 1, alpha: float = 0.1, beta: float = 0.5, smooth_iterations: int = 10) -> None:
    """
    Refines a mesh using trimesh for better detail.
    This involves subdivision to increase vertex count and smoothing.

    :param iterations: How many times to subdivide the mesh.
    :param alpha: Humphrey filter alpha value (controls shrinkage).
    :param beta: Humphrey filter beta value (controls smoothing).
    :param smooth_iterations: Number of smoothing iterations.
    """
    try:
        print(f"🐍 [Figurine Generator] Refining mesh at {ply_path} with {iterations} subdivision(s)...")
        mesh = trimesh.load(ply_path)

        # Subdivide the mesh. More iterations create significantly more vertices.
        for _ in range(iterations):
            mesh = mesh.subdivide()

        # Smooth the mesh to reduce jagged edges from subdivision.
        trimesh.smoothing.filter_humphrey(mesh, alpha=alpha, beta=beta, iterations=smooth_iterations)

        # Overwrite the original file with the refined mesh
        mesh.export(ply_path)
        print(f"✅ [Figurine Generator] Mesh refined and saved successfully.")
    except Exception as e:
        print(f"⚠️ [Figurine Generator] Could not refine mesh: {e}")

def _scale_mesh(ply_path: str, max_dimension_mm: float):
    """
    Scales a mesh to a maximum bounding box dimension.

    :param ply_path: Path to the PLY file.
    :param max_dimension_mm: The maximum size for the largest dimension in millimeters.
    """
    try:
        print(f"🐍 [Figurine Generator] Scaling mesh at {ply_path} to max {max_dimension_mm}mm...")
        mesh = trimesh.load(ply_path)

        # Get the current bounding box size
        current_max_dimension = np.max(mesh.extents)

        if current_max_dimension == 0:
            print("⚠️ [Figurine Generator] Cannot scale mesh with zero size.")
            return

        # Calculate the scaling factor
        scale_factor = max_dimension_mm / current_max_dimension

        # Apply the scaling transformation
        mesh.apply_scale(scale_factor)

        # Overwrite the original file with the scaled mesh
        mesh.export(ply_path)
        print(f"✅ [Figurine Generator] Mesh scaled successfully. New max dimension is approx {max_dimension_mm}mm.")
    except Exception as e:
        print(f"⚠️ [Figurine Generator] Could not scale mesh: {e}")

def generate_figurine(prompt: str, quality: str = "standard", output_dir: str = "Examples/generated_figurines") -> str:
    """
    Generates a 3D figurine model from a text prompt using different pipelines based on quality.
    - quality 'petit': Shap-E, fast preview, scaled to 25mm.
    - quality 'standard': Shap-E, fast preview.
    - quality 'detailed': Shap-E, higher quality.
    - quality 'ultra_realistic': TripoSR, highest quality.
    """
    initialize_pipelines()

    print(f"🐍 [Figurine Generator] Generating '{quality}' model for prompt: '{prompt}'...")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')[:30]
    filename = f"{timestamp}_{sanitized_prompt}_{quality}.ply"
    output_path = os.path.join(output_dir, filename)

    try:
        if quality == "ultra_realistic":
            if not pipe_tripo_sr:
                return "Error: TripoSR pipeline is not available. Check logs for details."

            print("🔥 [TripoSR] Generating with TripoSR...")
            mesh = pipe_tripo_sr(prompt, output_type="mesh").images[0]
            # TripoSR returns a dict with vertices and faces
            mesh_to_save = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
            mesh_to_save.export(output_path)

            # Apply more aggressive refinement for TripoSR
            _refine_mesh(output_path, iterations=2, alpha=0.05, beta=0.2)

        else:  # 'petit', 'standard', or 'detailed'
            if not pipe_shap_e:
                return "Error: Shap-E pipeline is not available. Check logs for details."

            if quality == "detailed":
                inference_steps, frame_size = 128, 512
            elif quality == "petit":
                inference_steps, frame_size = 32, 128  # Faster settings for small models
            else:  # standard
                inference_steps, frame_size = 64, 256

            print(f"🔷 [Shap-E] Generating with {inference_steps} steps...")
            mesh = pipe_shap_e(
                prompt,
                guidance_scale=15.0,
                num_inference_steps=inference_steps,
                frame_size=frame_size,
                output_type="mesh"
            ).images[0]
            export_to_ply(mesh, output_path)

            if quality == "detailed":
                _refine_mesh(output_path)

            if quality == "petit":
                _scale_mesh(output_path, max_dimension_mm=25.0)

        print(f"✅ [Figurine Generator] Model saved successfully to: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ [Figurine Generator] An error occurred during model generation: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"

if __name__ == '__main__':
    print("\n--- Running standalone test of figurine_generator.py ---")
    test_output_dir = "Examples/generated_figurines_test"
    print(f"Test output directory: '{test_output_dir}'")

    # Test 1: Standard Quality (Shap-E)
    test_prompt_standard = "a robot toy"
    print(f"\n[1] Testing Standard Quality (Shap-E)...")
    path_standard = generate_figurine(test_prompt_standard, quality="standard", output_dir=test_output_dir)
    if "Error" not in path_standard:
        print(f"    ✅ Standard test successful! Model saved at: {path_standard}")
    else:
        print(f"    ❌ Standard test failed. Reason: {path_standard}")

    # Test 2: Detailed Quality (Shap-E)
    test_prompt_detailed = "a detailed sports car"
    print(f"\n[2] Testing Detailed Quality (Shap-E)...")
    path_detailed = generate_figurine(test_prompt_detailed, quality="detailed", output_dir=test_output_dir)
    if "Error" not in path_detailed:
        print(f"    ✅ Detailed test successful! Model saved at: {path_detailed}")
    else:
        print(f"    ❌ Detailed test failed. Reason: {path_detailed}")

    # Test 3: Ultra-Realistic Quality (TripoSR)
    test_prompt_realistic = "a high-resolution DSLR camera"
    print(f"\n[3] Testing Ultra-Realistic Quality (TripoSR)...")
    path_realistic = generate_figurine(test_prompt_realistic, quality="ultra_realistic", output_dir=test_output_dir)
    if "Error" not in path_realistic:
        print(f"    ✅ Ultra-Realistic test successful! Model saved at: {path_realistic}")
    else:
        print(f"    ❌ Ultra-Realistic test failed. Reason: {path_realistic}")

    print("\n--- Standalone test finished ---")
    print(f"You can view the generated .ply files in the '{test_output_dir}' directory.")
