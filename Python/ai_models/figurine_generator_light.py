import os
import shutil
import random
from datetime import datetime

# --- Configuration ---
output_dir = "Examples/generated_figurines"
# Path to the directory containing placeholder models.
placeholder_dir = "Examples/generated_figurines"

def generate_figurine(prompt: str, quality: str = "standard") -> str:
    """
    (Light Version) Simulates the generation of a 3D figurine.
    Instead of a real model, it copies a *random* placeholder file to the output dir.
    This provides variety for UI/flow testing without heavy dependencies.
    """
    print(f"🐍 [Figurine Generator Light] Simulating '{quality}' model for prompt: '{prompt}'...")

    # --- Find available placeholder models ---
    try:
        available_models = [f for f in os.listdir(placeholder_dir) if f.startswith("placeholder_") and f.endswith(".ply")]
        if not available_models:
            raise FileNotFoundError("No placeholder models found in the directory.")

        # --- Select a random placeholder model ---
        selected_model_name = random.choice(available_models)
        placeholder_model_path = os.path.join(placeholder_dir, selected_model_name)
        print(f"🐍 [Figurine Generator Light] Selected random placeholder: {selected_model_name}")

    except FileNotFoundError as e:
        error_message = f"Error: Placeholder directory not found at '{placeholder_dir}' or it's empty. {e}"
        print(f"❌ {error_message}")
        return error_message
    except Exception as e:
        error_message = f"Error scanning for placeholder models: {e}"
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

    # --- Copy the selected random placeholder file ---
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
