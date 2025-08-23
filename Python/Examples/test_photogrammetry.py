import sys
import os

# Add the parent directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.image_to_3d import generate_3d_model_from_images

def test_photogrammetry():
    # Use the bridge test images
    test_image_dir = "../Examples/photogrammetry_test_images_bridge"
    test_images = []
    
    # Collect existing test images
    for i in range(5):
        image_path = os.path.join(test_image_dir, f"bridge_test_image_{i}.png")
        if os.path.exists(image_path):
            test_images.append(os.path.abspath(image_path))
    
    if not test_images:
        print("No test images found. Creating new ones...")
        from PIL import Image, ImageDraw
        os.makedirs(test_image_dir, exist_ok=True)
        
        for i in range(5):
            path = os.path.join(test_image_dir, f"bridge_test_image_{i}.png")
            img = Image.new("RGB", (200, 200), color=(137, 73, 109))
            d = ImageDraw.Draw(img)
            d.text((50, 50), f"Bridge Test {i+1}", fill=(255, 255, 0))
            img.save(path)
            test_images.append(os.path.abspath(path))
    
    print(f"Found {len(test_images)} test images")
    
    # Test with draft quality first (faster)
    print("\nTesting photogrammetry with 'Draft' quality...")
    result = generate_3d_model_from_images(
        test_images,
        quality="Draft",
        should_repair=True,
        target_size_mm=100.0
    )
    
    if "Error" not in result:
        print(f"\n✅ Success! Model generated at: {result}")
    else:
        print(f"\n❌ Failed: {result}")

if __name__ == "__main__":
    test_photogrammetry()
