import os
from datetime import datetime
from .opencv_photogrammetry import create_point_cloud
from .mesh_processor import repair_mesh, scale_mesh

# This script now uses OpenCV-based photogrammetry instead of Meshroom
# The main generation logic is in opencv_photogrammetry.py


def generate_3d_model_from_images(
    image_paths: list,
    output_dir: str = "Examples/generated_photogrammetry",
    quality: str = "balanced",
    detector_type: str = "SIFT",
    should_repair: bool = True,
    target_size_mm: float = 0.0,
    min_matches: int = 50,
    enhance_mesh: bool = True,
) -> str:
    """
    Generates a 3D model from a list of image files using enhanced OpenCV-based photogrammetry.
    
    Args:
        image_paths: A list of absolute paths to the input images.
        output_dir: The base directory where the final model will be saved.
        quality: Quality level - "fast", "balanced", "high"
        detector_type: Feature detector - "SIFT", "ORB", "AKAZE"
        should_repair: Boolean flag to enable/disable mesh repair.
        target_size_mm: The target size in mm for the model's longest dimension.
        min_matches: Minimum number of feature matches required between images
        enhance_mesh: Whether to apply AI mesh enhancement
        
    Returns:
        A string containing the path to the generated model file, or an error message.
    """
    if not isinstance(image_paths, list) or not image_paths:
        return "Error: Input must be a non-empty list of image paths."
    
    if len(image_paths) < 2:
        return "Error: At least 2 images are required for photogrammetry reconstruction."

    print(f"🔄 Processing {len(image_paths)} images...")
    print(f"⚙️  Quality: {quality}, Detector: {detector_type}, Min matches: {min_matches}")
    print(f"🛠️  Options: Repair={should_repair}, Target Size={target_size_mm}mm, Enhance={enhance_mesh}")

    # Validate image paths
    valid_images = []
    for img_path in image_paths:
        if os.path.exists(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            valid_images.append(img_path)
        else:
            print(f"⚠️  Skipping invalid image: {img_path}")
    
    if len(valid_images) < 2:
        return "Error: At least 2 valid images are required after filtering."
    
    print(f"✅ Using {len(valid_images)} valid images")

    # Create output directory with enhanced timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"reconstruction_{quality}_{detector_type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Generate initial point cloud with enhanced parameters
    point_cloud_path = os.path.join(run_dir, "point_cloud.ply")
    
    try:
        result = create_point_cloud(
            valid_images, 
            point_cloud_path, 
            detector_type=detector_type,
            quality=quality,
            min_matches=min_matches
        )
        
        if not result:
            return "Error: Failed to create point cloud from images. Try different quality settings or add more images."
            
        current_path = result
        print(f"📍 Point cloud created: {current_path}")
        
        # Apply mesh enhancement if requested
        if enhance_mesh:
            try:
                print("🎯 Applying AI mesh enhancement...")
                from .mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
                
                enhancer = MeshEnhancer(
                    MeshEnhancementConfig(
                        resolution_factor=1.2,
                        smoothness_weight=0.3,
                        detail_preservation=0.8
                    )
                )
                
                # Load and enhance the point cloud
                import trimesh
                mesh = trimesh.load(current_path)
                if hasattr(mesh, 'vertices') and len(mesh.vertices) > 100:
                    enhanced_mesh = enhancer.enhance_mesh(mesh)
                    enhanced_path = os.path.join(run_dir, "model_enhanced.ply")
                    enhanced_mesh.export(enhanced_path)
                    current_path = enhanced_path
                    print("✨ Mesh enhancement completed successfully")
                else:
                    print("⚠️  Mesh enhancement skipped - insufficient vertices")
                    
            except Exception as e:
                print(f"⚠️  Mesh enhancement failed: {e}")
                print("📋 Proceeding with original mesh...")
        
        # Post-process the mesh if needed
        if should_repair:
            repaired_path = os.path.join(run_dir, "model_repaired.obj")
            if repair_mesh(current_path, repaired_path):
                current_path = repaired_path
                print("🔧 Mesh repair completed successfully")
            else:
                print("⚠️  Mesh repair failed, proceeding with original mesh")
        
        if target_size_mm > 0:
            scaled_path = os.path.join(run_dir, "model_final.obj")
            if scale_mesh(current_path, scaled_path, target_size_mm):
                current_path = scaled_path
                print(f"📏 Mesh scaled to {target_size_mm}mm successfully")
            else:
                print("⚠️  Mesh scaling failed, using unscaled version")
        
        # Generate metadata
        file_size_mb = round(os.path.getsize(current_path) / (1024 * 1024), 2) if os.path.exists(current_path) else 0
        
        print(f"✅ Model generation complete!")
        print(f"📊 Final model: {current_path}")
        print(f"💾 File size: {file_size_mb} MB")
        
        return current_path
        
    except Exception as e:
        error_msg = f"Error during model generation: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Enhanced error categorization
        if "insufficient" in str(e).lower():
            suggestion = "Try using more images or reducing min_matches parameter"
        elif "memory" in str(e).lower():
            suggestion = "Try using 'fast' quality or fewer images"
        elif "feature" in str(e).lower():
            suggestion = "Try different detector_type (SIFT, ORB, or AKAZE)"
        else:
            suggestion = "Check image quality and ensure they show the same object from different angles"
            
        return f"{error_msg}. Suggestion: {suggestion}"


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
