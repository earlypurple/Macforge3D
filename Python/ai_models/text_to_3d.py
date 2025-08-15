import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_ply
from requests.exceptions import ConnectionError

import os
import json
from datetime import datetime

# --- Configuration ---
# Hugging Face model name for the Shap-E model.
model_name = "openai/shap-e"

# --- Global Variables ---
# We'll load the pipeline once and reuse it to save memory and time.
pipe = None
is_pipe_loaded = False
device = None


def initialize_pipeline() -> bool:
    """
    Initializes the Shap-E pipeline and moves it to the appropriate device.
    Returns True on success, False on failure.
    """
    global pipe, is_pipe_loaded, device

    if is_pipe_loaded:
        return True

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

    print(f"🐍 Using device: {device}")

    # --- Load the pipeline ---
    print(f"🐍 Loading Shap-E pipeline from '{model_name}'...")
    try:
        pipe = ShapEPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        print("✅ Pipeline loaded successfully.")
        is_pipe_loaded = True
        return True
    except ConnectionError:
        print(
            "❌ Network error: Failed to download the model. Please check your internet connection."
        )
        pipe = None
        is_pipe_loaded = False
        return False
    except Exception as e:
        print(f"❌ Failed to load the Shap-E pipeline: {e}")
        pipe = None
        is_pipe_loaded = False
        return False


def generate_3d_model(
    prompt: str,
    quality: str = "Balanced",
    output_dir: str = "Examples/generated_models",
) -> str:
    """
    Generates a 3D model from a text prompt.
    Returns a JSON string with the status and either a file path or an error message.
    """
    # --- Quality Settings ---
    quality_settings = {
        "Fast": {"steps": 32, "guidance": 10.0},
        "Balanced": {"steps": 64, "guidance": 15.0},
        "High Quality": {"steps": 96, "guidance": 20.0},
    }
    settings = quality_settings.get(quality, quality_settings["Balanced"])
    num_inference_steps = settings["steps"]
    guidance_scale = settings["guidance"]

    # --- Create a helper for JSON error responses ---
    def create_error_response(message: str) -> str:
        return json.dumps({"status": "error", "message": message})

    # --- Initialize pipeline ---
    if not is_pipe_loaded:
        if not initialize_pipeline():
            return create_error_response(
                "Failed to initialize the AI model. Check your internet connection and try again."
            )

    if not pipe:
        return create_error_response(
            "AI pipeline is not available. Please restart the application."
        )

    print(f"🐍 Generating 3D model for prompt: '{prompt}' with quality '{quality}'...")

    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate a unique filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_prompt = (
        "".join(c for c in prompt if c.isalnum() or c in (" ", "_"))
        .rstrip()
        .replace(" ", "_")[:30]
    )
    filename = f"{timestamp}_{quality}_{sanitized_prompt}.ply"
    output_path = os.path.join(output_dir, filename)

    # --- Run the model inference ---
    try:
        images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            frame_size=256,
            output_type="mesh",
        ).images

        if not images:
            raise ValueError("Model did not return any valid meshes.")

        mesh = images[0]
        export_to_ply(mesh, output_path)

        print(f"✅ Model saved successfully to: {output_path}")
        return json.dumps({"status": "success", "path": output_path})

    except Exception as e:
        error_message = f"An unexpected error occurred during model generation: {e}"
        print(f"❌ {error_message}")
        return create_error_response(error_message)


if __name__ == "__main__":
    print("\n--- Running standalone test of text_to_3d.py ---")
    test_prompt = "a robot wearing a cowboy hat"
    test_quality = "Fast"
    test_output_dir = "Examples/generated_models_test"
    print(f"Test prompt: '{test_prompt}'")
    print(f"Test quality: '{test_quality}'")
    print(f"Test output directory: '{test_output_dir}'")

    result_json = generate_3d_model(
        test_prompt, quality=test_quality, output_dir=test_output_dir
    )
    result = json.loads(result_json)

    if result.get("status") == "success":
        path = result.get("path")
        print(f"\n✅ Test successful! Model saved at: {path}")
        print(
            f"You can view the generated .ply file in the '{test_output_dir}' directory."
        )
    else:
        message = result.get("message")
        print(f"❌ Test failed. Reason: {message}")
