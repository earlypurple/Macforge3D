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

    if not os.path.exists(mesh_dir):
        print(f"❌ Texturing output directory not found: {mesh_dir}")
        return None

    potential_dirs = [d for d in os.listdir(mesh_dir) if os.path.isdir(os.path.join(mesh_dir, d))]
    if not potential_dirs:
        print(f"❌ No subdirectories found in Texturing cache: {mesh_dir}")
        return None

    latest_dir = sorted(potential_dirs, reverse=True)[0]
    texture_folder = os.path.join(mesh_dir, latest_dir)

    for file in os.listdir(texture_folder):
        if file.lower().endswith(".obj"):
            return os.path.join(texture_folder, file)

    print(f"❌ No .obj file found in the output directory: {texture_folder}")
    return None

def parse_meshroom_error(log: str) -> str:
    """
    Parses the Meshroom log to find common, user-friendly error messages.
    """
    if "No camera calibrated" in log or "Unable to calibrate cameras" in log:
        return "Error: Could not match features between images. Please ensure your photos have good overlap and are well-lit."
    if "Insufficient number of images" in log:
        return "Error: Not enough images to start the reconstruction. Please provide more images."
    if "FATAL" in log:
        # Find the line with the fatal error for more context
        for line in log.splitlines():
            if "FATAL" in line:
                return f"Error: A critical error occurred in Meshroom: {line}"
    return "" # Return empty if no specific error is found

def run_photogrammetry(
    image_paths: list,
    output_base_dir: str = "Examples/generated_photogrammetry",
    quality: str = "Default",
    should_repair: bool = True,
    target_size_mm: float = 0.0
) -> str:
    """
    Runs the Meshroom photogrammetry pipeline and optionally processes the output.

    Args:
        image_paths: A list of absolute paths to the input images.
        output_base_dir: The base directory to save the final model.
        quality: The desired quality level ("Draft", "Default", "High").
        should_repair: Boolean flag to enable/disable mesh repair.
        target_size_mm: The target size for the model's longest dimension.

    Returns:
        The path to the generated and processed 3D model file, or an error string.
    """
    if not image_paths:
        return "Error: No image paths provided."

    # --- Map quality setting to --scale parameter ---
    quality_map = {
        "Draft": "4",
        "Default": "2",
        "High": "1"
    }
    scale_factor = quality_map.get(quality, "2")
    print(f"🐍 Quality set to '{quality}'. Using scale factor: {scale_factor}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_base_dir, f"run_{timestamp}_{quality}")
    os.makedirs(run_output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_cache_dir:
        print(f"🐍 Using temporary cache directory: {temp_cache_dir}")

        command = [
            "meshroom_batch",
            "--input", *image_paths,
            "--output", run_output_dir,
            "--cache", temp_cache_dir,
            "--scale", scale_factor
        ]

        print(f"🐍 Executing command: {' '.join(command)}")

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False # Do not raise exception on non-zero exit codes
            )

            # Print stdout and stderr for debugging purposes
            print("--- Meshroom STDOUT ---")
            print(process.stdout)
            print("--- Meshroom STDERR ---")
            print(process.stderr)
            print("-----------------------")

            if process.returncode != 0:
                full_log = process.stdout + "\n" + process.stderr
                specific_error = parse_meshroom_error(full_log)
                if specific_error:
                    print(f"❌ Specific error identified: {specific_error}")
                    return specific_error

                error_message = f"Error: Meshroom process failed with return code {process.returncode}. Check console for logs."
                print(f"❌ {error_message}")
                return error_message

            print("✅ Meshroom process completed successfully.")

            raw_model_path = find_generated_model(temp_cache_dir)
            if not raw_model_path:
                return "Error: Could not find the generated model file in Meshroom's output. The reconstruction may have failed silently."

            model_dir = os.path.dirname(raw_model_path)
            for file in os.listdir(model_dir):
                shutil.copy(os.path.join(model_dir, file), run_output_dir)

            current_path = os.path.join(run_output_dir, os.path.basename(raw_model_path))
            print(f"✅ Raw model and assets copied to: {run_output_dir}")

            if should_repair:
                base_name = os.path.splitext(os.path.basename(current_path))[0]
                repaired_path = os.path.join(run_output_dir, f"{base_name}_repaired.obj")
                if repair_mesh(current_path, repaired_path):
                    current_path = repaired_path
                else:
                    print("⚠️  Mesh repair failed. Proceeding with the original model.")

            base_name = os.path.splitext(os.path.basename(current_path))[0]
            final_path = os.path.join(run_output_dir, f"{base_name}_processed.obj")

            if scale_mesh(current_path, final_path, target_size_mm):
                current_path = final_path
            else:
                print("⚠️  Mesh scaling failed. Saving the unscaled version.")
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
    print("\n--- Running standalone test of photogrammetry_pipeline.py ---")

    test_image_dir = "Examples/photogrammetry_test_images"
    os.makedirs(test_image_dir, exist_ok=True)

    test_images = []
    num_images = 5
    for i in range(num_images):
        path = os.path.join(test_image_dir, f"test_image_{i}.png")
        if not os.path.exists(path):
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (200, 200), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((50, 50), f"Image {i+1}", fill=(255, 255, 0))
            img.save(path)
        test_images.append(os.path.abspath(path))

    if len(test_images) != num_images:
        print("❌ Could not create or find all test images. Aborting test.")
    else:
        print(f"Found/created {len(test_images)} test images in {test_image_dir}")

        # --- Test with different quality settings ---
        for quality_level in ["Draft", "Default", "High"]:
            print(f"\n--- Testing photogrammetry pipeline with '{quality_level}' quality ---")
            result_path = run_photogrammetry(
                test_images,
                quality=quality_level,
                should_repair=True,
                target_size_mm=100.0
            )

            if result_path and "Error" not in result_path:
                print(f"\n✅ '{quality_level}' quality test successful! Model saved at: {result_path}")
                # Clean up the generated directory for the next run
                # shutil.rmtree(os.path.dirname(result_path))
            else:
                print(f"\n❌ '{quality_level}' quality test failed. Reason: {result_path}")
                print("    Please ensure Meshroom is installed and 'meshroom_batch' is in your PATH.")
                break # Stop testing if one level fails
