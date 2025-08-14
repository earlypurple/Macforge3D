from .photogrammetry_pipeline import run_photogrammetry
import os
from PIL import Image, ImageDraw

# This script now acts as a bridge to the photogrammetry pipeline.
# The main generation logic is in `photogrammetry_pipeline.py`.

def generate_3d_model_from_images(image_paths: list, output_dir: str = "Examples/generated_photogrammetry") -> str:
    """
    Generates a 3D model from a list of image files using the photogrammetry pipeline.
    This function is the main entry point called from the Swift application.

    Args:
        image_paths: A list of absolute paths to the input images.
        output_dir: The base directory where the final model and its assets will be saved.

    Returns:
        A string containing the path to the generated model file, or an error message.
    """
    print(f"🐍 Bridge script 'image_to_3d.py' received request for {len(image_paths)} images.")

    if not isinstance(image_paths, list) or not image_paths:
        return "Error: Input must be a list of image paths."

    # Delegate the heavy lifting to the specialized photogrammetry script.
    result = run_photogrammetry(image_paths, output_base_dir=output_dir)

    print(f"🐍 Photogrammetry pipeline finished. Result: {result}")
    return result

if __name__ == '__main__':
    # This block allows for testing the script directly from the command line.
    print("\n--- Running standalone test of image_to_3d.py bridge ---")

    # Create a dummy set of images for testing.
    test_image_dir = "Examples/photogrammetry_test_images_bridge"
    os.makedirs(test_image_dir, exist_ok=True)

    test_images = []
    num_images = 5
    for i in range(num_images):
        path = os.path.join(test_image_dir, f"bridge_test_image_{i}.png")
        if not os.path.exists(path):
            try:
                img = Image.new('RGB', (200, 200), color=(137, 73, 109))
                d = ImageDraw.Draw(img)
                d.text((50, 50), f"Bridge Test {i+1}", fill=(255, 255, 0))
                img.save(path)
                test_images.append(os.path.abspath(path))
            except Exception as e:
                print(f"Could not create dummy image {path}: {e}")
        else:
            test_images.append(os.path.abspath(path))

    if len(test_images) != num_images:
        print("❌ Could not create or find all test images. Aborting test.")
    else:
        print(f"Found/created {len(test_images)} test images for bridge test.")

        # Call the main function of this script
        path = generate_3d_model_from_images(test_images)

        if path and "Error" not in path:
            print(f"\n✅ Bridge test successful! Model saved at: {path}")
            print(f"This confirms that 'image_to_3d.py' can correctly call the 'photogrammetry_pipeline'.")
        else:
            print(f"\n❌ Bridge test failed. Reason: {path}")
            print("Please ensure Meshroom is installed and 'meshroom_batch' is in your PATH.")
