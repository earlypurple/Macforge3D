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
    style: str = "realistic",
    enhance_mesh: bool = True,
    frame_size: int = 256,
) -> str:
    """
    Generates a 3D model from a text prompt with enhanced quality options.
    
    Args:
        prompt: Text description of the 3D model
        quality: Quality level - "Fast", "Balanced", "High Quality", "Ultra"
        output_dir: Directory to save the generated model
        style: Generation style - "realistic", "stylized", "artistic", "geometric"
        enhance_mesh: Whether to apply AI mesh enhancement
        frame_size: Resolution of the generated mesh (128, 256, 512)
        
    Returns:
        JSON string with status and either file path or error message
    """
    # --- Enhanced Quality Settings ---
    quality_settings = {
        "Fast": {"steps": 24, "guidance": 8.0, "eta": 0.0},
        "Balanced": {"steps": 48, "guidance": 12.0, "eta": 0.1},
        "High Quality": {"steps": 80, "guidance": 18.0, "eta": 0.2},
        "Ultra": {"steps": 120, "guidance": 25.0, "eta": 0.3},
    }
    
    # Style-specific guidance adjustments
    style_modifiers = {
        "realistic": {"guidance_boost": 1.0, "steps_boost": 1.0},
        "stylized": {"guidance_boost": 1.2, "steps_boost": 0.9},
        "artistic": {"guidance_boost": 1.5, "steps_boost": 1.1},
        "geometric": {"guidance_boost": 0.8, "steps_boost": 0.8},
    }
    
    settings = quality_settings.get(quality, quality_settings["Balanced"])
    style_mod = style_modifiers.get(style, style_modifiers["realistic"])
    
    num_inference_steps = int(settings["steps"] * style_mod["steps_boost"])
    guidance_scale = settings["guidance"] * style_mod["guidance_boost"]
    eta = settings.get("eta", 0.1)

    # --- Input validation ---
    if not prompt or len(prompt.strip()) < 3:
        return json.dumps({"status": "error", "message": "Prompt must be at least 3 characters long"})
    
    if frame_size not in [128, 256, 512]:
        frame_size = 256  # Default fallback
        
    # --- Create enhanced helper for JSON responses ---
    def create_response(status: str, message: str = "", path: str = "", metadata: dict = None) -> str:
        response = {"status": status}
        if message:
            response["message"] = message
        if path:
            response["path"] = path
        if metadata:
            response["metadata"] = metadata
        return json.dumps(response)

    # --- Initialize pipeline with retry logic ---
    max_retries = 3
    for attempt in range(max_retries):
        if not is_pipe_loaded:
            if not initialize_pipeline():
                if attempt == max_retries - 1:
                    return create_response("error", 
                        "Failed to initialize AI model after multiple attempts. Check your internet connection.")
                print(f"Retry {attempt + 1}/{max_retries} initializing pipeline...")
                continue

        if not pipe:
            return create_response("error", "AI pipeline is not available. Please restart the application.")
        break

    print(f"🐍 Generating 3D model for prompt: '{prompt}'")
    print(f"⚙️  Quality: {quality}, Style: {style}, Frame size: {frame_size}x{frame_size}")
    print(f"📊 Steps: {num_inference_steps}, Guidance: {guidance_scale:.1f}, Eta: {eta}")

    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate enhanced filename with metadata ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_prompt = (
        "".join(c for c in prompt if c.isalnum() or c in (" ", "_"))
        .rstrip()
        .replace(" ", "_")[:30]
    )
    filename = f"{timestamp}_{quality}_{style}_{frame_size}_{sanitized_prompt}.ply"
    output_path = os.path.join(output_dir, filename)

    # --- Enhanced prompt preprocessing ---
    enhanced_prompt = prompt
    if style == "realistic":
        enhanced_prompt = f"highly detailed, photorealistic {prompt}, 8k quality"
    elif style == "stylized":
        enhanced_prompt = f"stylized, clean design {prompt}, smooth surfaces"
    elif style == "artistic":
        enhanced_prompt = f"artistic, creative interpretation of {prompt}, unique style"
    elif style == "geometric":
        enhanced_prompt = f"geometric, minimalist {prompt}, clean lines"

    # --- Run the model inference with enhanced parameters ---
    try:
        print(f"🚀 Starting generation with enhanced prompt: '{enhanced_prompt[:50]}...'")
        
        # Generate with enhanced parameters
        images = pipe(
            enhanced_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            frame_size=frame_size,
            output_type="mesh",
            eta=eta,
        ).images

        if not images or len(images) == 0:
            raise ValueError("Model did not return any valid meshes.")

        mesh = images[0]
        
        # --- Apply mesh enhancement if requested ---
        if enhance_mesh:
            try:
                print("🎯 Applying AI mesh enhancement...")
                # Import mesh enhancer here to avoid circular imports
                from .mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
                
                enhancer = MeshEnhancer(
                    MeshEnhancementConfig(
                        resolution_factor=1.3,
                        smoothness_weight=0.2,
                        detail_preservation=0.9
                    )
                )
                
                # Convert to trimesh for enhancement
                if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                    import trimesh
                    temp_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                    enhanced_mesh = enhancer.enhance_mesh(temp_mesh)
                    # Convert back to expected format for export
                    mesh.vertices = enhanced_mesh.vertices
                    mesh.faces = enhanced_mesh.faces
                    print("✨ Mesh enhancement completed successfully")
                else:
                    print("⚠️  Mesh enhancement skipped - incompatible mesh format")
                    
            except Exception as e:
                print(f"⚠️  Mesh enhancement failed: {e}")
                print("📋 Proceeding with original mesh...")

        # --- Export the final mesh ---
        export_to_ply(mesh, output_path)
        
        # --- Generate metadata ---
        metadata = {
            "generation_time": datetime.now().isoformat(),
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "quality_settings": {
                "quality": quality,
                "style": style,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "frame_size": frame_size,
                "eta": eta
            },
            "mesh_enhanced": enhance_mesh,
            "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2) if os.path.exists(output_path) else 0
        }

        print(f"✅ Model saved successfully to: {output_path}")
        print(f"📊 File size: {metadata['file_size_mb']} MB")
        
        return create_response("success", "3D model generated successfully", output_path, metadata)

    except Exception as e:
        error_message = f"Generation failed: {str(e)}"
        print(f"❌ {error_message}")
        
        # Enhanced error categorization
        if "CUDA" in str(e) or "memory" in str(e).lower():
            suggestion = "Try reducing frame_size or using 'Fast' quality setting"
        elif "connection" in str(e).lower() or "network" in str(e).lower():
            suggestion = "Check your internet connection and try again"
        elif "timeout" in str(e).lower():
            suggestion = "Generation timed out. Try using 'Fast' quality or a simpler prompt"
        else:
            suggestion = "Try a different prompt or quality setting"
            
        return create_response("error", f"{error_message}. Suggestion: {suggestion}")


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
