import subprocess
import os
import tempfile
import shutil
from datetime import datetime
from .mesh_processor import repair_mesh, scale_mesh

def find_generated_model(output_dir):
    """Finds the most likely final model file from the Meshroom output."""
    # Meshroom typically outputs the final textured mesh here.
    mesh_dir = os.path.join(output_dir, "MeshroomCache", "Texturing")

    # The output folder name can vary based on Meshroom version and graph.
    # We look for a folder inside Texturing, then find an .obj file.
    if not os.path.exists(mesh_dir):
        print(f"❌ Texturing output directory not found: {mesh_dir}")
        return None

    potential_dirs = [d for d in os.listdir(mesh_dir) if os.path.isdir(os.path.join(mesh_dir, d))]
    if not potential_dirs:
        print(f"❌ No subdirectories found in Texturing cache: {mesh_dir}")
        return None

    # Sort to get the latest one if there are multiple
    latest_dir = sorted(potential_dirs, reverse=True)[0]
    texture_folder = os.path.join(mesh_dir, latest_dir)

    for file in os.listdir(texture_folder):
        if file.lower().endswith(".obj"):
            return os.path.join(texture_folder, file)

    print(f"❌ No .obj file found in the output directory: {texture_folder}")
    return None


def run_photogrammetry(
    image_paths: list,
    output_base_dir: str = "Examples/generated_photogrammetry",
    should_repair: bool = True,
    target_size_mm: float = 0.0
) -> str:
    """
    Runs the Meshroom photogrammetry pipeline and optionally processes the output.

    Args:
        image_paths: A list of absolute paths to the input images.
        output_base_dir: The base directory to save the final model.
        should_repair: Boolean flag to enable/disable mesh repair.
        target_size_mm: The target size for the model's longest dimension.

    Returns:
        The path to the generated and processed 3D model file, or an error string.
    """
    if not image_paths:
        return "Error: No image paths provided."

    # --- Create a unique output directory for this run ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Use a temporary directory for the Meshroom cache to keep things clean.
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        print(f"🐍 Using temporary cache directory: {temp_cache_dir}")

        # --- Construct the command for meshroom_batch ---
        # Ensure meshroom_batch is in the system's PATH.
        command = [
            "meshroom_batch",
            "--input", *image_paths,
            "--output", run_output_dir,
            "--cache", temp_cache_dir
        ]

        print(f"🐍 Executing command: {' '.join(command)}")

        try:
            # --- Execute the command ---
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            # --- Stream output in real-time ---
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    print(f"[Meshroom]: {line.strip()}")

            process.wait() # Wait for the process to complete

            if process.returncode != 0:
                error_message = f"Error: Meshroom process failed with return code {process.returncode}."
                print(f"❌ {error_message}")
                return error_message

            print("✅ Meshroom process completed successfully.")

            # --- Find the generated model file in the output directory ---
            raw_model_path = find_generated_model(temp_cache_dir)

            if not raw_model_path:
                return "Error: Could not find the generated model file in Meshroom's output."

            # --- Copy all generated assets (model, mtl, textures) to the final output directory ---
            model_dir = os.path.dirname(raw_model_path)
            for file in os.listdir(model_dir):
                shutil.copy(os.path.join(model_dir, file), run_output_dir)

            # This is the path to the raw model inside our persistent output folder
            current_path = os.path.join(run_output_dir, os.path.basename(raw_model_path))

            print(f"✅ Raw model and assets copied to: {run_output_dir}")

            # --- Post-processing ---

            # Stage 1: Repair
            if should_repair:
                base_name = os.path.splitext(os.path.basename(current_path))[0]
                repaired_path = os.path.join(run_output_dir, f"{base_name}_repaired.obj")
                if repair_mesh(current_path, repaired_path):
                    # If repair is successful, the next stage uses the repaired model
                    current_path = repaired_path
                else:
                    print("⚠️  Mesh repair failed. Proceeding with the original model.")

            # Stage 2: Scale
            # The final model will be named based on the input, but with a suffix
            base_name = os.path.splitext(os.path.basename(current_path))[0]
            final_path = os.path.join(run_output_dir, f"{base_name}_processed.obj")

            if scale_mesh(current_path, final_path, target_size_mm):
                current_path = final_path
            else:
                print("⚠️  Mesh scaling failed. Saving the unscaled version.")
                # Ensure the file is at the expected final path
                if current_path != final_path:
                    shutil.move(current_path, final_path)
                    mtl_src = current_path.replace('.obj', '.mtl')
                    if os.path.exists(mtl_src):
                        shutil.move(mtl_src, final_path.replace('.obj', '.mtl'))
                current_path = final_path

            print(f"✅ Post-processing complete. Final model is at: {current_path}")
            return current_path

        except FileNotFoundError:
            error_message = "Error: 'meshroom_batch' command not found. Is Meshroom installed and in the system's PATH?"
            print(f"❌ {error_message}")
            return error_message
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(f"❌ {error_message}")
            return error_message

if __name__ == '__main__':
    # This block allows for testing the script directly from the command line.
    print("\n--- Running standalone test of photogrammetry_pipeline.py ---")

    # Create a dummy set of images for testing.
    test_image_dir = "Examples/photogrammetry_test_images"
    os.makedirs(test_image_dir, exist_ok=True)

    test_images = []
    for i in range(5):
        path = os.path.join(test_image_dir, f"test_image_{i}.png")
        if not os.path.exists(path):
            try:
                # Create simple dummy images. For a real test, these would be photos of an object.
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (200, 200), color = (73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text((50, 50), f"Image {i+1}", fill=(255, 255, 0))
                img.save(path)
                test_images.append(os.path.abspath(path))
            except Exception as e:
                print(f"Could not create dummy image {path}: {e}")
        else:
             test_images.append(os.path.abspath(path))

    if not test_images:
        print("❌ Could not create or find test images. Aborting test.")
    else:
        print(f"Found/created {len(test_images)} test images in {test_image_dir}")
        print("\n--- Testing photogrammetry pipeline with post-processing ---")
        result_path = run_photogrammetry(test_images, should_repair=True, target_size_mm=200.0)

        if "Error" in result_path:
            print(f"\n❌ Photogrammetry test failed: {result_path}")
            print("Please ensure Meshroom is installed and 'meshroom_batch' is in your PATH.")
        else:
            print(f"\n✅ Photogrammetry test successful! Model saved at: {result_path}")
