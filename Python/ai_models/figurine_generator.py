import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_ply
import trimesh
import numpy as np
import os
from datetime import datetime
from typing import Optional
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# --- Configuration ---
shap_e_model_name = "openai/shap-e"
tripo_sr_model_name = "stabilityai/TripoSR"

# --- Global Variables ---
pipe_shap_e = None
pipe_tripo_sr = None
device = None
torch_dtype = None


def initialize_pipelines():
    """Initializes all 3D generation pipelines and moves them to the appropriate device."""
    global pipe_shap_e, pipe_tripo_sr, device, torch_dtype, shap_e_model_name, tripo_sr_model_name

    if pipe_shap_e and pipe_tripo_sr:
        return

    # --- Setup device ---
    if not device:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            torch_dtype = torch.float16
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            torch_dtype = torch.float16
        else:
            device = torch.device("cpu")
            torch_dtype = torch.float32
        print(
            f"🐍 [Figurine Generator] Using device: {device} with dtype: {torch_dtype}"
        )

    # --- Load Shap-E Pipeline ---
    if not pipe_shap_e:
        print(
            f"🐍 [Figurine Generator] Loading Shap-E pipeline from '{shap_e_model_name}'..."
        )
        try:
            pipe_shap_e = ShapEPipeline.from_pretrained(
                shap_e_model_name, torch_dtype=torch_dtype
            )
            pipe_shap_e = pipe_shap_e.to(device)
            print("✅ [Figurine Generator] Shap-E pipeline loaded successfully.")
        except Exception as e:
            print(f"❌ [Figurine Generator] Failed to load the Shap-E pipeline: {e}")
            pipe_shap_e = None

    # --- Load TripoSR Model ---
    if not pipe_tripo_sr:
        print(
            f"🐍 [Figurine Generator] Loading TripoSR model from '{tripo_sr_model_name}'..."
        )
        try:
            pipe_tripo_sr = TSR.from_pretrained(
                tripo_sr_model_name,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            pipe_tripo_sr.renderer.set_chunk_size(8192)
            pipe_tripo_sr.to(device)
            print("✅ [Figurine Generator] TripoSR model loaded successfully.")
        except Exception as e:
            print(f"❌ [Figurine Generator] Failed to load the TripoSR model: {e}")
            pipe_tripo_sr = None


def _refine_mesh(
    ply_path: str,
    iterations: int = 1,
    alpha: float = 0.1,
    beta: float = 0.5,
    smooth_iterations: int = 10,
) -> None:
    """
    Refines a mesh using trimesh for better detail.
    This involves subdivision to increase vertex count and smoothing.

    :param iterations: How many times to subdivide the mesh.
    :param alpha: Humphrey filter alpha value (controls shrinkage).
    :param beta: Humphrey filter beta value (controls smoothing).
    :param smooth_iterations: Number of smoothing iterations.
    """
    try:
        print(
            f"🐍 [Figurine Generator] Refining mesh at {ply_path} with {iterations} subdivision(s)..."
        )
        mesh = trimesh.load(ply_path)

        # Subdivide the mesh. More iterations create significantly more vertices.
        for _ in range(iterations):
            mesh = mesh.subdivide()

        # Smooth the mesh to reduce jagged edges from subdivision.
        trimesh.smoothing.filter_humphrey(
            mesh, alpha=alpha, beta=beta, iterations=smooth_iterations
        )

        # Overwrite the original file with the refined mesh
        mesh.export(ply_path)
        print(f"✅ [Figurine Generator] Mesh refined and saved successfully.")
    except Exception as e:
        print(f"⚠️ [Figurine Generator] Could not refine mesh: {e}")


def _apply_advanced_mesh_optimization(mesh):
    """
    Apply advanced mesh optimization techniques that don't require neural networks.
    
    Args:
        mesh: trimesh.Trimesh object to optimize
        
    Returns:
        Optimized trimesh.Trimesh object
    """
    try:
        # Remove duplicate vertices
        mesh.merge_vertices()
        
        # Remove degenerate faces
        mesh.remove_degenerate_faces()
        
        # Remove duplicate faces
        mesh.remove_duplicate_faces()
        
        # Fix normals if needed
        if not mesh.is_winding_consistent:
            mesh.fix_normals()
        
        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()
        
        # Apply edge-based smoothing if mesh is not too large
        if len(mesh.vertices) < 50000:
            try:
                # Simple Laplacian smoothing
                mesh = mesh.smoothed()
            except Exception:
                pass  # Skip if smoothing fails
        
        return mesh
        
    except Exception as e:
        print(f"   Warning: Advanced optimization failed: {e}")
        return mesh


def _apply_maximum_enhancement(ply_path: str) -> None:
    """
    Applies maximum quality enhancement techniques to a mesh.
    
    This function combines multiple enhancement approaches:
    - Multiple subdivision iterations for increased geometry density
    - Advanced smoothing with optimized parameters
    - Neural mesh enhancement if available
    - Quality analysis and optimization
    
    :param ply_path: Path to the PLY file to enhance
    """
    try:
        print("🔥 [Max Enhancement] Starting maximum quality enhancement pipeline...")
        
        # Load the mesh
        mesh = trimesh.load(ply_path)
        original_vertex_count = len(mesh.vertices)
        print(f"📊 [Max Enhancement] Original mesh: {original_vertex_count} vertices, {len(mesh.faces)} faces")
        
        # Phase 1: Advanced subdivision for geometry density
        print("🔧 [Max Enhancement] Phase 1: Advanced subdivision...")
        for i in range(3):  # More aggressive subdivision
            mesh = mesh.subdivide()
            print(f"   Subdivision {i+1}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Phase 2: Basic mesh optimization
        print("🔧 [Max Enhancement] Phase 2: Basic mesh optimization...")
        mesh = _apply_advanced_mesh_optimization(mesh)
        
        # Phase 3: Mesh quality improvement
        print("🔧 [Max Enhancement] Phase 3: Mesh quality optimization...")
        
        # Remove degenerate faces and merge close vertices
        original_faces = len(mesh.faces)
        mesh.remove_degenerate_faces()
        if len(mesh.faces) < original_faces:
            print(f"   Removed {original_faces - len(mesh.faces)} degenerate faces")
        
        original_vertices = len(mesh.vertices)
        mesh.merge_vertices()
        if len(mesh.vertices) < original_vertices:
            print(f"   Merged {original_vertices - len(mesh.vertices)} close vertices")
        
        # Fix mesh orientation
        if not mesh.is_winding_consistent:
            mesh.fix_normals()
            print("   Fixed mesh normal orientation")
        
        # Phase 4: Advanced smoothing with multiple techniques
        print("🔧 [Max Enhancement] Phase 4: Advanced smoothing...")
        
        # Apply Humphrey filter with optimized parameters for maximum quality
        try:
            trimesh.smoothing.filter_humphrey(
                mesh, alpha=0.05, beta=0.2, iterations=15
            )
            print("   Applied Humphrey smoothing filter")
        except Exception as e:
            print(f"   Warning: Humphrey smoothing failed: {e}")
        
        # Apply Laplacian smoothing
        try:
            mesh = mesh.smoothed()
            print("   Applied Laplacian smoothing")
        except Exception as e:
            print(f"   Warning: Laplacian smoothing failed: {e}")
        
        # Phase 5: Advanced mesh quality enhancement using mesh_processor
        print("🔧 [Max Enhancement] Phase 5: Advanced quality enhancement...")
        try:
            from .mesh_processor import enhance_mesh_quality
            
            enhanced_mesh, enhancement_stats = enhance_mesh_quality(
                mesh,
                smooth_iterations=3,
                fix_orientation=True,
                remove_degenerate=True,
                merge_close_vertices=True
            )
            
            if enhancement_stats["success"]:
                mesh = enhanced_mesh
                print("   Applied advanced mesh quality enhancement")
                for improvement in enhancement_stats["improvements"]:
                    print(f"     - {improvement}")
            else:
                print("   Warning: Advanced quality enhancement failed")
                
        except ImportError:
            print("   Advanced quality enhancement not available (mesh_processor import failed)")
        except Exception as e:
            print(f"   Warning: Advanced quality enhancement failed: {e}")
        
        # Phase 6: Attempt neural enhancement if mesh_enhancer is available
        print("🔧 [Max Enhancement] Phase 6: Neural enhancement attempt...")
        try:
            from .mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
            
            config = MeshEnhancementConfig(
                resolution_factor=2.0,
                smoothness_weight=0.3,
                detail_preservation=0.8,
                max_points=150000
            )
            
            enhancer = MeshEnhancer(config)
            enhanced_mesh = enhancer.enhance_mesh(mesh, return_original_scale=True)
            mesh = enhanced_mesh
            print("   Applied neural mesh enhancement")
            
        except ImportError:
            print("   Neural enhancement not available (mesh_enhancer import failed)")
        except Exception as e:
            print(f"   Warning: Neural enhancement failed: {e}")
        
        # Phase 7: Final quality validation and export
        print("🔧 [Max Enhancement] Phase 7: Final optimization and export...")
        
        # Ensure mesh is watertight if possible
        if not mesh.is_watertight:
            try:
                mesh.fill_holes()
                if mesh.is_watertight:
                    print("   Made mesh watertight")
            except Exception as e:
                print(f"   Warning: Could not make mesh watertight: {e}")
        
        # Final quality report
        final_vertex_count = len(mesh.vertices)
        final_face_count = len(mesh.faces)
        enhancement_ratio = final_vertex_count / original_vertex_count
        
        print(f"📊 [Max Enhancement] Final mesh: {final_vertex_count} vertices, {final_face_count} faces")
        print(f"📊 [Max Enhancement] Enhancement ratio: {enhancement_ratio:.2f}x geometry density")
        print(f"📊 [Max Enhancement] Watertight: {mesh.is_watertight}")
        print(f"📊 [Max Enhancement] Volume: {mesh.volume:.6f}")
        
        # Export the enhanced mesh
        mesh.export(ply_path)
        print("✅ [Max Enhancement] Maximum quality enhancement completed successfully!")
        
    except Exception as e:
        print(f"❌ [Max Enhancement] Enhancement failed: {e}")
        import traceback
        traceback.print_exc()


def _scale_mesh(ply_path: str, max_dimension_mm: float):
    """
    Scales a mesh to a maximum bounding box dimension.

    :param ply_path: Path to the PLY file.
    :param max_dimension_mm: The maximum size for the largest dimension in millimeters.
    """
    try:
        print(
            f"🐍 [Figurine Generator] Scaling mesh at {ply_path} to max {max_dimension_mm}mm..."
        )
        mesh = trimesh.load(ply_path)

        # Get the current bounding box size
        current_max_dimension = np.max(mesh.extents)

        if current_max_dimension == 0:
            print("⚠️ [Figurine Generator] Cannot scale mesh with zero size.")
            return

        # Calculate the scaling factor
        scale_factor = max_dimension_mm / current_max_dimension

        # Apply the scaling transformation
        mesh.apply_scale(scale_factor)

        # Overwrite the original file with the scaled mesh
        mesh.export(ply_path)
        print(
            f"✅ [Figurine Generator] Mesh scaled successfully. New max dimension is approx {max_dimension_mm}mm."
        )
    except Exception as e:
        print(f"⚠️ [Figurine Generator] Could not scale mesh: {e}")


def generate_figurine(
    prompt: str,
    quality: str = "standard",
    output_dir: str = "Examples/generated_figurines",
    image_path: Optional[str] = None,
) -> str:
    """
    Generates a 3D figurine model.
    - quality 'petit', 'standard', 'detailed': Uses Shap-E for text-to-3D generation.
    - quality 'ultra_realistic': Uses TripoSR for image-to-3D generation.
    - quality 'max': Uses the highest quality settings with maximum enhancement.
    :param prompt: The text prompt for text-to-3D models.
    :param quality: The quality setting.
    :param output_dir: The directory to save the output file.
    :param image_path: The path to the input image for image-to-3D models.
    """
    initialize_pipelines()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if quality == "ultra_realistic":
        if image_path is None:
            return "Error: Image path is required for ultra-realistic quality."
        print(
            f"🐍 [Figurine Generator] Generating '{quality}' model for image: '{image_path}'..."
        )
        sanitized_prompt = os.path.splitext(os.path.basename(image_path))[0]
    else:
        print(
            f"🐍 [Figurine Generator] Generating '{quality}' model for prompt: '{prompt}'..."
        )
        sanitized_prompt = (
            "".join(c for c in prompt if c.isalnum() or c in (" ", "_"))
            .rstrip()
            .replace(" ", "_")[:30]
        )

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{timestamp}_{sanitized_prompt}_{quality}.ply"
    output_path = os.path.join(output_dir, filename)

    try:
        if quality == "ultra_realistic":
            if not pipe_tripo_sr:
                return "Error: TripoSR model is not available. Check logs for details."

            print("🔥 [TripoSR] Generating with TripoSR...")
            assert image_path is not None
            image = Image.open(image_path)
            # You may want to remove the background here if the input image has one
            # image = remove_background(image)
            with torch.no_grad():
                scene_codes = pipe_tripo_sr([image], device=device)

            meshes = pipe_tripo_sr.extract_mesh(scene_codes)
            meshes[0].export(output_path)

            # Apply more aggressive refinement for TripoSR
            _refine_mesh(output_path, iterations=2, alpha=0.05, beta=0.2)

        else:  # 'petit', 'standard', or 'detailed'
            if not pipe_shap_e:
                return (
                    "Error: Shap-E pipeline is not available. Check logs for details."
                )

            if quality == "max":
                inference_steps, frame_size = 256, 1024
            elif quality == "detailed":
                inference_steps, frame_size = 128, 512
            elif quality == "petit":
                inference_steps, frame_size = (32, 128)
            else:  # standard
                inference_steps, frame_size = 64, 256

            print(f"🔷 [Shap-E] Generating with {inference_steps} steps...")
            mesh = pipe_shap_e(
                prompt,
                guidance_scale=20.0 if quality == "max" else 15.0,
                num_inference_steps=inference_steps,
                frame_size=frame_size,
                output_type="mesh",
            ).images[0]
            export_to_ply(mesh, output_path)

            # Apply quality-specific enhancements
            if quality == "max":
                print("🚀 [Figurine Generator] Applying maximum quality enhancements...")
                _apply_maximum_enhancement(output_path)
            elif quality == "detailed":
                _refine_mesh(output_path)

            if quality == "petit":
                _scale_mesh(output_path, max_dimension_mm=25.0)

        print(f"✅ [Figurine Generator] Model saved successfully to: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ [Figurine Generator] An error occurred during model generation: {e}")
        import traceback

        traceback.print_exc()
        return f"Error: {e}"


if __name__ == "__main__":
    print("\n--- Running standalone test of figurine_generator.py ---")
    test_output_dir = "Examples/generated_figurines_test"
    print(f"Test output directory: '{test_output_dir}'")

    # Test 1: Standard Quality (Shap-E)
    test_prompt_standard = "a robot toy"
    print(f"\n[1] Testing Standard Quality (Shap-E)...")
    path_standard = generate_figurine(
        test_prompt_standard, quality="standard", output_dir=test_output_dir
    )
    if "Error" not in path_standard:
        print(f"    ✅ Standard test successful! Model saved at: {path_standard}")
    else:
        print(f"    ❌ Standard test failed. Reason: {path_standard}")

    # Test 2: Detailed Quality (Shap-E)
    test_prompt_detailed = "a detailed sports car"
    print(f"\n[2] Testing Detailed Quality (Shap-E)...")
    path_detailed = generate_figurine(
        test_prompt_detailed, quality="detailed", output_dir=test_output_dir
    )
    if "Error" not in path_detailed:
        print(f"    ✅ Detailed test successful! Model saved at: {path_detailed}")
    else:
        print(f"    ❌ Detailed test failed. Reason: {path_detailed}")

    # Test 3: Maximum Quality (Shap-E with full enhancement)
    test_prompt_max = "a detailed dragon figurine"
    print(f"\n[3] Testing Maximum Quality (Shap-E with full enhancement)...")
    path_max = generate_figurine(
        test_prompt_max, quality="max", output_dir=test_output_dir
    )
    if "Error" not in path_max:
        print(f"    ✅ Maximum quality test successful! Model saved at: {path_max}")
    else:
        print(f"    ❌ Maximum quality test failed. Reason: {path_max}")

    # Test 4: Ultra-Realistic Quality (TripoSR)
    test_image_path = (
        "Examples/photogrammetry_test_images_bridge/bridge_test_image_0.png"
    )
    print(
        f"\n[4] Testing Ultra-Realistic Quality (TripoSR) with image {test_image_path}..."
    )
    path_realistic = generate_figurine(
        "a bridge",
        quality="ultra_realistic",
        output_dir=test_output_dir,
        image_path=test_image_path,
    )
    if "Error" not in path_realistic:
        print(
            f"    ✅ Ultra-Realistic test successful! Model saved at: {path_realistic}"
        )
    else:
        print(f"    ❌ Ultra-Realistic test failed. Reason: {path_realistic}")

    print("\n--- Standalone test finished ---")
    print(
        f"You can view the generated .ply files in the '{test_output_dir}' directory."
    )
