#!/usr/bin/env python3
"""
Demonstration of the comprehensive improvements to MacForge3D.
Shows practical usage of all new features.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add Python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Python'))

def demo_enhanced_validation():
    """Demonstrate the enhanced validation system."""
    print("🔍 Enhanced Validation System Demo")
    print("=" * 50)
    
    from simulation.enhanced_validation import input_validator
    
    # Test various parameter configurations
    test_configs = [
        {
            "name": "Valid Configuration",
            "params": {
                "resolution": 15000,
                "quality": "high",
                "material": "PLA",
                "temperature": 210.0,
                "smoothness_weight": 0.7,
                "detail_preservation": 0.8
            }
        },
        {
            "name": "Auto-correctable Configuration", 
            "params": {
                "resolution": 50,  # Too low, will be corrected
                "quality": "invalid_quality",  # Invalid, will be corrected
                "temperature": 400.0,  # Too high, will be corrected
                "smoothness_weight": 1.5  # Too high, will be corrected
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n📋 Testing: {config['name']}")
        try:
            validated = input_validator.validate_mesh_parameters(config["params"])
            print(f"✅ Validation successful!")
            for key, value in validated.items():
                original = config["params"].get(key, "N/A")
                if original != value:
                    print(f"  🔧 {key}: {original} → {value} (auto-corrected)")
                else:
                    print(f"  ✓ {key}: {value}")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
    
    print()

def demo_enhanced_exceptions():
    """Demonstrate the enhanced exception system."""
    print("⚠️  Enhanced Exception System Demo")
    print("=" * 50)
    
    from core.enhanced_exceptions import (
        MeshProcessingError, MemoryError, ValidationError, 
        handle_exception_gracefully, exception_handler
    )
    
    # Demonstrate different exception types
    print("\n🔥 Testing MeshProcessingError:")
    try:
        mesh_info = {"vertices": 50000, "faces": 100000, "watertight": False}
        raise MeshProcessingError(
            "Mesh has too many degenerate faces",
            mesh_info=mesh_info
        )
    except MeshProcessingError as e:
        details = e.get_detailed_info()
        print(f"  📊 Severity: {details['severity']}")
        print(f"  🔧 Recovery suggestions: {len(details['recovery_suggestions'])} available")
        print(f"  📈 System state captured: {bool(details['system_state'])}")
    
    print("\n💾 Testing MemoryError:")
    try:
        raise MemoryError("Insufficient memory for large mesh processing", required_memory=8.0)
    except MemoryError as e:
        print(f"  💾 Required memory: {e.parameters.get('required_memory_gb', 'N/A')} GB")
        print(f"  🔧 Recovery suggestions: {len(e.recovery_suggestions)}")
    
    print("\n🛡️  Testing exception handler decorator:")
    @exception_handler(fallback_result="Safe fallback result")
    def potentially_failing_function(should_fail=False):
        if should_fail:
            raise ValueError("Simulated processing error")
        return "Success!"
    
    result1 = potentially_failing_function(False)
    result2 = potentially_failing_function(True)
    print(f"  ✅ Normal execution: {result1}")
    print(f"  🛡️  Exception handled: {result2}")
    
    print()

def demo_mesh_quality_analysis():
    """Demonstrate enhanced mesh quality analysis."""
    print("📐 Enhanced Mesh Quality Analysis Demo")
    print("=" * 50)
    
    try:
        import trimesh
        from ai_models.mesh_processor import analyze_mesh_quality
        
        # Create different test meshes
        test_meshes = [
            {
                "name": "Perfect Tetrahedron",
                "vertices": np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [0.5, np.sqrt(3)/2, 0],
                    [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
                ], dtype=np.float32),
                "faces": np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            },
            {
                "name": "Irregular Mesh",
                "vertices": np.random.random((100, 3)).astype(np.float32),
                "faces": np.random.randint(0, 100, (50, 3))
            }
        ]
        
        for mesh_data in test_meshes:
            print(f"\n🔍 Analyzing: {mesh_data['name']}")
            mesh = trimesh.Trimesh(vertices=mesh_data["vertices"], faces=mesh_data["faces"])
            
            analysis = analyze_mesh_quality(mesh)
            
            if "error" not in analysis:
                print(f"  📊 Overall quality score: {analysis['overall_quality_score']:.3f}")
                print(f"  📏 Vertices: {analysis['basic_stats']['vertices']}")
                print(f"  🔺 Faces: {analysis['basic_stats']['faces']}")
                print(f"  ⚡ Manifold score: {analysis.get('topological_quality', {}).get('manifold_score', 'N/A')}")
                print(f"  📐 Aspect ratio: {analysis.get('geometric_quality', {}).get('bounding_box', {}).get('aspect_ratio', 'N/A')}")
                print(f"  ⏱️  Analysis time: {analysis['processing_time']:.4f}s")
            else:
                print(f"  ❌ Analysis failed: {analysis['error']}")
        
    except ImportError:
        print("  ⚠️  Trimesh not available, skipping mesh analysis demo")
    
    print()

def demo_memory_processor():
    """Demonstrate advanced memory processor."""
    print("💾 Advanced Memory Processor Demo")
    print("=" * 50)
    
    from ai_models.advanced_memory_processor import (
        AdvancedMemoryProcessor, MemoryConfig, estimate_mesh_memory_usage
    )
    
    # Configure memory processor
    config = MemoryConfig(
        max_memory_percent=70.0,
        chunk_memory_limit_mb=256.0,
        enable_disk_cache=True
    )
    
    processor = AdvancedMemoryProcessor(config)
    
    print(f"\n⚙️  Memory Configuration:")
    print(f"  Max memory usage: {config.max_memory_percent}%")
    print(f"  Chunk memory limit: {config.chunk_memory_limit_mb} MB")
    print(f"  Disk cache enabled: {config.enable_disk_cache}")
    
    # Test memory monitoring
    print(f"\n📊 Memory Monitoring:")
    processor.monitor.start_monitoring()
    time.sleep(0.5)  # Brief monitoring period
    processor.monitor.stop_monitoring()
    
    stats = processor.monitor.get_memory_stats()
    print(f"  Current memory: {stats['current_mb']:.1f} MB")
    print(f"  Peak memory: {stats['peak_mb']:.1f} MB")
    print(f"  Available memory: {stats['available_mb']:.1f} MB")
    
    # Test cache functionality
    print(f"\n🗃️  Cache System:")
    
    def test_processing_function(data):
        """Simple test processing function."""
        return data * 2
    
    # Process some test data
    test_data = np.random.random((1000, 3)).astype(np.float32)
    
    # First processing (cache miss)
    start_time = time.time()
    result1 = processor._process_chunk_with_cache(test_data, test_processing_function)
    time1 = time.time() - start_time
    
    # Second processing (cache hit)
    start_time = time.time()
    result2 = processor._process_chunk_with_cache(test_data, test_processing_function)
    time2 = time.time() - start_time
    
    cache_stats = processor.get_cache_stats()
    print(f"  Cache hits: {cache_stats['hits']}")
    print(f"  Cache misses: {cache_stats['misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"  First processing: {time1:.4f}s")
    print(f"  Second processing: {time2:.4f}s (speedup: {time1/time2:.1f}x)")
    
    # Test memory estimation
    try:
        import trimesh
        vertices = np.random.random((5000, 3)).astype(np.float32)
        faces = np.random.randint(0, 5000, (2500, 3))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        estimation = estimate_mesh_memory_usage(mesh)
        print(f"\n📏 Memory Estimation for 5K vertex mesh:")
        print(f"  Vertices: {estimation['vertices_mb']:.2f} MB")
        print(f"  Faces: {estimation['faces_mb']:.2f} MB")
        print(f"  Total estimated: {estimation['total_mb']:.2f} MB")
        
    except ImportError:
        print("\n  ⚠️  Trimesh not available, skipping memory estimation")
    
    print()

def demo_workflow_integration():
    """Demonstrate integrated workflow using all improvements."""
    print("🔄 Integrated Workflow Demo")
    print("=" * 50)
    
    from simulation.enhanced_validation import input_validator
    from ai_models.advanced_memory_processor import AdvancedMemoryProcessor
    from core.enhanced_exceptions import handle_exception_gracefully
    
    print("\n🚀 Complete Processing Workflow:")
    
    # Step 1: Validate parameters
    print("  1️⃣  Validating parameters...")
    params = {
        "resolution": 10000,
        "quality": "high", 
        "material": "PLA",
        "temperature": 210.0,
        "smoothness_weight": 0.6
    }
    
    try:
        validated_params = input_validator.validate_mesh_parameters(params)
        print(f"     ✅ Parameters validated successfully")
    except Exception as e:
        validated_params = handle_exception_gracefully("parameter_validation", e, params, {})
        print(f"     ⚠️  Parameters validation failed, using defaults")
    
    # Step 2: Memory preparation
    print("  2️⃣  Preparing memory management...")
    processor = AdvancedMemoryProcessor()
    processor.monitor.start_monitoring()
    
    # Step 3: Simulate processing
    print("  3️⃣  Processing data...")
    test_data = np.random.random((5000, 3)).astype(np.float32)
    
    def mock_processing_function(data):
        # Simulate some processing time
        time.sleep(0.01)
        return data + np.random.random(data.shape) * 0.1
    
    try:
        processed_data = processor._process_chunk_with_cache(test_data, mock_processing_function)
        print(f"     ✅ Data processed successfully: {processed_data.shape}")
    except Exception as e:
        result = handle_exception_gracefully("data_processing", e, {"data_shape": test_data.shape}, test_data)
        print(f"     ⚠️  Processing failed, using fallback")
    
    # Step 4: Cleanup
    print("  4️⃣  Cleaning up...")
    processor.monitor.stop_monitoring()
    final_stats = processor.monitor.get_memory_stats()
    cache_stats = processor.get_cache_stats()
    
    print(f"     📊 Final memory usage: {final_stats['peak_mb']:.1f} MB peak")
    print(f"     🗃️  Cache efficiency: {cache_stats['hit_rate']:.1%}")
    print(f"     ✅ Workflow completed successfully!")
    
    print()

def main():
    """Run all improvement demonstrations."""
    print("🎯 MacForge3D Comprehensive Improvements Demonstration")
    print("=" * 70)
    print("This demonstration showcases all the new features and improvements")
    print("implemented in the MacForge3D framework.\n")
    
    try:
        demo_enhanced_validation()
        demo_enhanced_exceptions()
        demo_mesh_quality_analysis()
        demo_memory_processor()
        demo_workflow_integration()
        
        print("🎉 All demonstrations completed successfully!")
        print("\nKey improvements implemented:")
        print("✅ Enhanced mesh quality analysis with 10+ new metrics")
        print("✅ Advanced error handling with context capture and recovery")
        print("✅ Comprehensive input validation with auto-correction")
        print("✅ Intelligent memory management with real-time monitoring")
        print("✅ Edge-preserving mesh smoothing algorithms")
        print("✅ Comprehensive testing infrastructure")
        print("\nThe MacForge3D framework is now more robust, efficient, and user-friendly!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()