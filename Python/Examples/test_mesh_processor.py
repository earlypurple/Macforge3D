import sys
import os

# Add the parent directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.mesh_processor import repair_mesh, scale_mesh

def test_mesh_repair():
    input_path = "../Examples/mesh_processor_test/broken_cube.obj"
    output_path = "../Examples/mesh_processor_test/repaired_cube_test.obj"
    
    print(f"Testing mesh repair...")
    success = repair_mesh(input_path, output_path)
    print(f"Mesh repair {'succeeded' if success else 'failed'}")

def test_mesh_scaling():
    input_path = "../Examples/mesh_processor_test/repaired_cube.obj"
    output_path = "../Examples/mesh_processor_test/scaled_cube_test.obj"
    target_size_mm = 50.0  # Scale to 50mm
    
    print(f"Testing mesh scaling...")
    success = scale_mesh(input_path, output_path, target_size_mm)
    print(f"Mesh scaling {'succeeded' if success else 'failed'}")

if __name__ == "__main__":
    test_mesh_repair()
    test_mesh_scaling()
