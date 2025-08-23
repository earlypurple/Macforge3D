import os
from datetime import datetime
from .opencv_photogrammetry import create_point_cloud
from .mesh_processor import repair_mesh, scale_mesh

# This script now uses OpenCV-based photogrammetry instead of Meshroom
# The main generation logic is in opencv_photogrammetry.py


def generate_3d_model_from_images(
    image_paths: list,
    output_dir: str = "Examples/generated_photogrammetry",
    quality: str = "Default",
    should_repair: bool = True,
    target_size_mm: float = 0.0,
) -> str:
    """
    Generates a 3D model from a list of image files using OpenCV-based photogrammetry.
    This function is the main entry point called from the Swift application.
    
    Args:
        image_paths: A list of absolute paths to the input images.
        output_dir: The base directory where the final model will be saved.
        quality: The desired quality level (affects point cloud density).
        should_repair: Boolean flag to enable/disable mesh repair.
        target_size_mm: The target size in mm for the model's longest dimension.
        
    Returns:
        A string containing the path to the generated model file, or an error message.
    """
    if not isinstance(image_paths, list) or not image_paths:
        return "Error: Input must be a non-empty list of image paths."

    print(f"🔄 Processing {len(image_paths)} images...")
    print(f"⚙️  Options: Quality='{quality}', Repair={should_repair}, Target Size={target_size_mm}mm")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"reconstruction_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Generate initial point cloud
    point_cloud_path = os.path.join(run_dir, "point_cloud.ply")
    
    try:
        result = create_point_cloud(image_paths, point_cloud_path)
        if not result:
            return "Error: Failed to create point cloud from images."
            
        current_path = result
        
        # Post-process the mesh if needed
        if should_repair:
            repaired_path = os.path.join(run_dir, "model_repaired.obj")
            if repair_mesh(current_path, repaired_path):
                current_path = repaired_path
            else:
                print("⚠️  Mesh repair failed, proceeding with original mesh")
        
        if target_size_mm > 0:
            scaled_path = os.path.join(run_dir, "model_final.obj")
            if scale_mesh(current_path, scaled_path, target_size_mm):
                current_path = scaled_path
            else:
                print("⚠️  Mesh scaling failed, using unscaled version")
        
        print(f"✅ Model generation complete! Result saved at: {current_path}")
        return current_path
        
    except Exception as e:
        error_msg = f"Error during model generation: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg


if __name__ == "__main__":
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
                img = Image.new("RGB", (200, 200), color=(137, 73, 109))
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

        # Call the main function of this script with a specific quality
        print("\n--- Testing with 'Draft' quality ---")
        path = generate_3d_model_from_images(
            test_images, quality="Draft", should_repair=True, target_size_mm=150.0
        )

        if path and "Error" not in path:
            print(
                f"\n✅ Bridge test with 'Draft' quality successful! Model saved at: {path}"
            )
        else:
            print(f"\n❌ Bridge test with 'Draft' quality failed. Reason: {path}")
            print(
                "   Please ensure Meshroom is installed and 'meshroom_batch' is in your PATH."
            )

        # Call the main function of this script with default quality
        print("\n--- Testing with 'Default' quality and no post-processing ---")
        path_no_pp = generate_3d_model_from_images(
            test_images, quality="Default", should_repair=False, target_size_mm=0
        )

        if path_no_pp and "Error" not in path_no_pp:
            print(
                f"\n✅ Bridge test with 'Default' quality successful! Model saved at: {path_no_pp}"
            )
        else:
            print(
                f"\n❌ Bridge test with 'Default' quality failed. Reason: {path_no_pp}"
            )
            print(
                "   Please ensure Meshroom is installed and 'meshroom_batch' is in your PATH."
            )
