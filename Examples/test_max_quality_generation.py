#!/usr/bin/env python3
"""
Example script demonstrating the new maximum quality figurine generation.

This script shows how to use the new "max" quality level that provides:
- Highest resolution generation (256 inference steps, 1024 frame size)
- Advanced mesh subdivision for increased geometry density
- Multiple smoothing techniques
- Neural enhancement (when available)
- Comprehensive quality optimization pipeline

Usage:
    python Examples/test_max_quality_generation.py
"""

import sys
import os

# Add Python directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(os.path.dirname(script_dir), 'Python')
sys.path.insert(0, python_dir)

def main():
    """Demonstrate maximum quality figurine generation."""
    print("🎯 Maximum Quality Figurine Generation Demo")
    print("=" * 50)
    
    try:
        from ai_models.figurine_generator import generate_figurine
        
        # Create output directory
        output_dir = "Examples/max_quality_demo_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test prompts for different complexity levels
        test_prompts = [
            "a simple geometric cube",
            "a detailed dragon figurine",
            "an ornate chess piece knight",
            "a complex mechanical robot"
        ]
        
        print("\n🚀 Starting maximum quality generation tests...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[{i}] Testing with prompt: '{prompt}'")
            print("-" * 40)
            
            try:
                result_path = generate_figurine(
                    prompt=prompt,
                    quality="max",
                    output_dir=output_dir
                )
                
                if "Error" not in result_path:
                    print(f"✅ Generation successful!")
                    print(f"📄 Output: {result_path}")
                    
                    # Show file info if available
                    if os.path.exists(result_path):
                        file_size = os.path.getsize(result_path)
                        print(f"📊 File size: {file_size:,} bytes")
                    
                else:
                    print(f"❌ Generation failed: {result_path}")
                    
            except Exception as e:
                print(f"❌ Error during generation: {e}")
        
        print(f"\n📁 All outputs saved to: {output_dir}")
        
    except ImportError as e:
        print(f"\n⚠️  This demo requires full dependencies to be installed.")
        print(f"Import error: {e}")
        print("\nTo run this demo, install the required packages:")
        print("  pip install torch diffusers trimesh")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def show_quality_comparison():
    """Show comparison between different quality levels."""
    print("\n📊 Quality Level Comparison")
    print("=" * 40)
    
    quality_info = {
        "petit": {
            "inference_steps": 32,
            "frame_size": 128,
            "enhancements": ["Basic scaling to 25mm"],
            "use_case": "Small figurines, quick prototypes"
        },
        "standard": {
            "inference_steps": 64,
            "frame_size": 256,
            "enhancements": ["Standard generation"],
            "use_case": "General purpose figurines"
        },
        "detailed": {
            "inference_steps": 128,
            "frame_size": 512,
            "enhancements": ["Mesh refinement"],
            "use_case": "Higher detail requirements"
        },
        "max": {
            "inference_steps": 256,
            "frame_size": 1024,
            "enhancements": [
                "Advanced subdivision (3 iterations)",
                "Advanced mesh optimization",
                "Multiple smoothing techniques",
                "Quality enhancement pipeline",
                "Neural enhancement (when available)",
                "Comprehensive validation"
            ],
            "use_case": "Professional quality, maximum detail"
        },
        "ultra_realistic": {
            "model": "TripoSR",
            "input": "Image-to-3D",
            "enhancements": ["Aggressive refinement"],
            "use_case": "Photorealistic models from images"
        }
    }
    
    for quality, info in quality_info.items():
        print(f"\n🔹 {quality.upper()}")
        if "inference_steps" in info:
            print(f"   Steps: {info['inference_steps']}")
            print(f"   Frame Size: {info['frame_size']}")
        if "model" in info:
            print(f"   Model: {info['model']}")
        if "input" in info:
            print(f"   Input: {info['input']}")
        print(f"   Enhancements:")
        for enhancement in info["enhancements"]:
            print(f"     • {enhancement}")
        print(f"   Use Case: {info['use_case']}")

if __name__ == "__main__":
    show_quality_comparison()
    main()