import torch
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_ply, load_image

import os
from datetime import datetime
from PIL import Image

# --- Configuration ---
# Hugging Face model name for the Shap-E image-to-image model.
model_name = "openai/shap-e-img2img"

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
    print(f"🐍 Loading Shap-E Img2Img pipeline from '{model_name}'...")
    try:
        pipe = ShapEImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        print("✅ Pipeline loaded successfully.")
        is_pipe_loaded = True
    except Exception as e:
        print(f"❌ Failed to load the Shap-E pipeline: {e}")
        pipe = None
        is_pipe_loaded = False

def generate_3d_model_from_image(image_path: str, output_dir: str = "Examples/generated_models") -> str:
    """
    Generates a 3D model from an image file using the Shap-E model.
    If the model generation fails, it returns a string containing an error message.
    """
    # Initialize the pipeline on the first call.
    if not is_pipe_loaded:
        initialize_pipeline()

    if not pipe:
        return "Error: Shap-E pipeline is not available. Check logs for details."

    print(f"🐍 Generating 3D model for image: '{image_path}'...")

    # --- Load the input image ---
    try:
        input_image = load_image(image_path)
        print(f"✅ Image loaded successfully from: {image_path}")
    except Exception as e:
        error_message = f"Error: Failed to load image at '{image_path}'. Reason: {e}"
        print(f"❌ {error_message}")
        return error_message


    # --- Create output directory if it doesn't exist ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate a unique filename to avoid overwriting ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize the image filename to create a valid, short filename.
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    sanitized_filename = "".join(c for c in base_filename if c.isalnum() or c in (' ', '_')).rstrip()
    sanitized_filename = sanitized_filename.replace(' ', '_')[:30]
    filename = f"{timestamp}_{sanitized_filename}.ply"
    output_path = os.path.join(output_dir, filename)

    # --- Run the model inference ---
    try:
        # The guidance_scale and num_inference_steps are key parameters to tune quality vs. speed.
        images = pipe(
            input_image,
            guidance_scale=3.0,
            num_inference_steps=64,
            frame_size=256,
            output_type="mesh"  # Request a mesh output directly
        ).images

        if not images:
            raise ValueError("Model did not return any images/meshes.")

        mesh = images[0]

        # --- Save the generated mesh to a .ply file ---
        export_to_ply(mesh, output_path)

        print(f"✅ Model saved successfully to: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ An error occurred during model generation: {e}")
        return f"Error: {e}"

if __name__ == '__main__':
    # This block allows for testing the script directly from the command line.
    # Example: python Python/ai_models/image_to_3d.py
    print("\n--- Running standalone test of image_to_3d.py ---")

    # Create a dummy image for testing if one doesn't exist.
    test_image_path = "Examples/test_image.png"
    if not os.path.exists(test_image_path):
        print(f"Creating a dummy test image at: {test_image_path}")
        try:
            os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
            dummy_image = Image.new('RGB', (100, 100), color = 'red')
            dummy_image.save(test_image_path)
            print("✅ Dummy image created.")
        except Exception as e:
            print(f"❌ Failed to create dummy image: {e}")
            test_image_path = None

    if test_image_path:
        test_output_dir = "Examples/generated_models_test"
        print(f"Test image: '{test_image_path}'")
        print(f"Test output directory: '{test_output_dir}'")

        path = generate_3d_model_from_image(test_image_path, output_dir=test_output_dir)

        if path and "Error" not in path:
            print(f"\n✅ Test successful! Model saved at: {path}")
            print(f"You can view the generated .ply file in the '{test_output_dir}' directory using a 3D viewer like MeshLab or Blender.")
        else:
            print(f"\n❌ Test failed. Reason: {path}")
    else:
        print("\n❌ Test skipped because no test image was available.")
