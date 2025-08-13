import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_ply
import trimesh
import numpy as np

import os
from datetime import datetime

# --- Configuration ---
model_name = "openai/shap-e"

# --- Global Variables ---
pipe = None
is_pipe_loaded = False
device = None

def initialize_pipeline():
    """Initializes the Shap-E pipeline and moves it to the appropriate device."""
    global pipe, is_pipe_loaded, device

    if is_pipe_loaded:
        return

    # --- Setup device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch_dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32

    print(f"🐍 [Figurine Generator] Using device: {device}")

    # --- Load the pipeline ---
    print(f"🐍 [Figurine Generator] Loading Shap-E pipeline from '{model_name}'...")
    try:
        pipe = ShapEPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        print("✅ [Figurine Generator] Pipeline loaded successfully.")
        is_pipe_loaded = True
    except Exception as e:
        print(f"❌ [Figurine Generator] Failed to load the Shap-E pipeline: {e}")
        pipe = None
        is_pipe_loaded = False

def _refine_mesh(ply_path: str) -> None:
    """
    Refines a mesh using trimesh for better detail.
    This involves subdivision to increase vertex count and smoothing.
    """
    try:
        print(f"🐍 [Figurine Generator] Refining mesh at {ply_path}...")
        mesh = trimesh.load(ply_path)

        # Subdivide the mesh. This splits each triangle into smaller ones.
        # The number of iterations determines how many new vertices are created.
        # One iteration is often a good balance.
        mesh = mesh.subdivide()

        # Smooth the mesh to reduce jagged edges from subdivision.
        trimesh.smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.5, iterations=10)

        # Overwrite the original file with the refined mesh
        mesh.export(ply_path)
        print(f"✅ [Figurine Generator] Mesh refined and saved successfully.")
    except Exception as e:
        print(f"⚠️ [Figurine Generator] Could not refine mesh: {e}")
        # We don't re-raise the error, as refinement is an enhancement, not a critical step.

def generate_figurine(prompt: str, quality: str = "standard", output_dir: str = "Examples/generated_figurines") -> str:
    """
    Generates a 3D figurine model from a text prompt.
    - quality 'standard': Faster, good for previews.
    - quality 'ultra_detailed': Slower, higher detail, with mesh refinement.
    """
    if not is_pipe_loaded:
        initialize_pipeline()

    if not pipe:
        return "Error: Shap-E pipeline is not available. Check logs for details."

    print(f"🐍 [Figurine Generator] Generating '{quality}' model for prompt: '{prompt}'...")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
    sanitized_prompt = sanitized_prompt.replace(' ', '_')[:30]
    filename = f"{timestamp}_{sanitized_prompt}_{quality}.ply"
    output_path = os.path.join(output_dir, filename)

    # --- Set parameters based on quality ---
    if quality == "ultra_detailed":
        inference_steps = 128
        frame_size = 512
        guidance_scale = 15.0 # This is generally a good value
    else: # standard
        inference_steps = 64
        frame_size = 256
        guidance_scale = 15.0

    try:
        images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            frame_size=frame_size,
            output_type="mesh"
        ).images

        if not images:
            raise ValueError("Model did not return any meshes.")

        mesh = images[0]
        export_to_ply(mesh, output_path)

        # --- Apply post-processing for ultra-detailed models ---
        if quality == "ultra_detailed":
            _refine_mesh(output_path)

        print(f"✅ [Figurine Generator] Model saved successfully to: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ [Figurine Generator] An error occurred during model generation: {e}")
        return f"Error: {e}"

if __name__ == '__main__':
    print("\n--- Running standalone test of figurine_generator.py ---")
    test_output_dir = "Examples/generated_figurines_test"
    print(f"Test output directory: '{test_output_dir}'")

    # Test 1: Standard Quality
    test_prompt_standard = "a cute cat figurine"
    print(f"\n[1] Testing Standard Quality...")
    print(f"    Prompt: '{test_prompt_standard}'")
    path_standard = generate_figurine(test_prompt_standard, quality="standard", output_dir=test_output_dir)
    if "Error" not in path_standard:
        print(f"    ✅ Standard test successful! Model saved at: {path_standard}")
    else:
        print(f"    ❌ Standard test failed. Reason: {path_standard}")

    # Test 2: Ultra-Detailed Quality
    test_prompt_detailed = "an epic knight on a horse, detailed armor"
    print(f"\n[2] Testing Ultra-Detailed Quality...")
    print(f"    Prompt: '{test_prompt_detailed}'")
    path_detailed = generate_figurine(test_prompt_detailed, quality="ultra_detailed", output_dir=test_output_dir)
    if "Error" not in path_detailed:
        print(f"    ✅ Ultra-Detailed test successful! Model saved at: {path_detailed}")
    else:
        print(f"    ❌ Ultra-Detailed test failed. Reason: {path_detailed}")

    print("\n--- Standalone test finished ---")
    print(f"You can view the generated .ply files in the '{test_output_dir}' directory.")
