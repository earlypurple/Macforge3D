import os
import shutil
from datetime import datetime

# --- Configuration ---
output_dir = "Examples/generated_figurines"
# Path to the placeholder model. This is relative to the project root.
placeholder_model_path = "MacForge3D/Ressource/Models/placeholder_figurine.ply"

def generate_figurine(prompt: str, quality: str = "standard") -> str:
    """
    (Light Version) Simulates the generation of a 3D figurine.
    Instead of running a model, it copies a placeholder file to the output directory.
    This allows for testing the UI and application flow without heavy dependencies.
    """
    print(f"🐍 [Figurine Generator Light] Simulating '{quality}' model for prompt: '{prompt}'...")

    # --- Check if the placeholder file exists ---
    if not os.path.exists(placeholder_model_path):
        error_message = f"Error: Placeholder model not found at '{placeholder_model_path}'"
        print(f"❌ {error_message}")
        return error_message

    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate a unique filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
    sanitized_prompt = sanitized_prompt.replace(' ', '_')[:30]
    filename = f"{timestamp}_{sanitized_prompt}_{quality}_light.ply"
    output_path = os.path.join(output_dir, filename)

    # --- Copy the placeholder file ---
    try:
        shutil.copy(placeholder_model_path, output_path)
        print(f"✅ [Figurine Generator Light] Placeholder model copied successfully to: {output_path}")
        return output_path
    except Exception as e:
        error_message = f"Error: Failed to copy placeholder model: {e}"
        print(f"❌ {error_message}")
        return error_message

if __name__ == '__main__':
    print("\n--- Running standalone test of figurine_generator_light.py ---")

    test_prompt = "a simulated figurine"
    print(f"Test prompt: '{test_prompt}'")

    path = generate_figurine(test_prompt)

    if path and "Error" not in path:
        print(f"\n✅ Test successful! Simulated model saved at: {path}")
        print("This file is a copy of the placeholder model.")
    else:
        print(f"\n❌ Test failed. Reason: {path}")
