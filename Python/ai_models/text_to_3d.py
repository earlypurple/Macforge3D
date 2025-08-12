import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_ply

import os
from datetime import datetime

# --- Configuration ---
# Directory to save the generated models. This path is relative to the project root.
output_dir = "Examples/generated_models"
# Hugging Face model name for the Shap-E model.
model_name = "openai/shap-e"

# --- Global Variables ---
# We'll load the pipeline once and reuse it to save memory and time.
pipe = None
is_pipe_loaded = False
device = None

def initialize_pipeline():
    """Initializes the Shap-E pipeline and moves it to the appropriate device."""
    global pipe, is_pipe_loaded, device

    if is_pipe_loaded:
        return

    # --- Setup device ---
    # Check for Apple's Metal Performance Shaders (MPS) for GPU acceleration on Mac.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch_dtype = torch.float16  # Use float16 for memory efficiency on MPS
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32  # CPU works better with float32

    print(f"🐍 Using device: {device}")

    # --- Load the pipeline ---
    # This will download the model on the first run and cache it in ~/.cache/huggingface/hub.
    print(f"🐍 Loading Shap-E pipeline from '{model_name}'...")
    try:
        pipe = ShapEPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        print("✅ Pipeline loaded successfully.")
        is_pipe_loaded = True
    except Exception as e:
        print(f"❌ Failed to load the Shap-E pipeline: {e}")
        # This could be due to no internet connection, or an invalid model name.
        pipe = None
        is_pipe_loaded = False

def generate_3d_model(prompt: str) -> str:
    """
    Generates a 3D model from a text prompt using the Shap-E model.
    If the model generation fails, it returns a string containing an error message.
    """
    # Initialize the pipeline on the first call.
    if not is_pipe_loaded:
        initialize_pipeline()

    if not pipe:
        return "Error: Shap-E pipeline is not available. Check logs for details."

    print(f"🐍 Generating 3D model for prompt: '{prompt}'...")

    # --- Create output directory if it doesn't exist ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate a unique filename to avoid overwriting ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize the prompt to create a valid, short filename.
    sanitized_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
    sanitized_prompt = sanitized_prompt.replace(' ', '_')[:30]
    filename = f"{timestamp}_{sanitized_prompt}.ply"
    output_path = os.path.join(output_dir, filename)

    # --- Run the model inference ---
    try:
        # The guidance_scale and num_inference_steps are key parameters to tune quality vs. speed.
        images = pipe(
            prompt,
            guidance_scale=15.0,
            num_inference_steps=64,  # Higher steps can improve quality but take longer.
            frame_size=256,        # The size of the latent images, affects detail.
            output_type="mesh"
        ).images

        # The pipeline returns a list of meshes; for Shap-E, it's typically one.
        if not images:
            raise ValueError("Model did not return any images/meshes.")

        mesh = images[0]

        # --- Save the generated mesh to a .ply file ---
        # We use the export_to_ply utility function from the diffusers library.
        export_to_ply(mesh, output_path)

        print(f"✅ Model saved successfully to: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ An error occurred during model generation: {e}")
        # Return the error message to be displayed in the UI.
        return f"Error: {e}"

if __name__ == '__main__':
    # This block allows for testing the script directly from the command line.
    # Example: python Python/ai_models/text_to_3d.py
    print("\n--- Running standalone test of text_to_3d.py ---")
    test_prompt = "a robot wearing a cowboy hat"
    print(f"Test prompt: '{test_prompt}'")

    path = generate_3d_model(test_prompt)

    if path and "Error" not in path:
        print(f"\n✅ Test successful! Model saved at: {path}")
        # Provide instructions for viewing the model
        print("You can view the generated .ply file using a 3D viewer like MeshLab or Blender.")
    else:
        print(f"\n❌ Test failed. Reason: {path}")
