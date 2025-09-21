#!/usr/bin/env python3
"""
Script de test pour vérifier que tous les modules Python se chargent correctement.
"""

import sys
import importlib
import traceback
from pathlib import Path

# Ajouter le répertoire Python au path
python_dir = Path(__file__).parent
sys.path.insert(0, str(python_dir))

# Liste des modules à tester
modules_to_test = [
    'ai_models.image_to_3d',
    'ai_models.figurine_generator',
    'ai_models.performance_optimizer',
    'ai_models.cluster_manager',
    'ai_models.text_effects',
    'ai_models.mesh_processor',
    'ai_models.opencv_photogrammetry',
    'ai_models.test_smart_cache',
    'ai_models.smart_cache',
    'ai_models.text_to_mesh',
    'ai_models.text_to_mesh_optimized',
    'ai_models.cache_extensions',
    'ai_models.auto_optimizer',
    'ai_models.custom_compression',
]

def test_module_import(module_name: str) -> bool:
    """Teste l'import d'un module."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: ImportError - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name}: {type(e).__name__} - {e}")
        return False

def main():
    """Fonction principale."""
    print("🔍 Test d'import des modules Python...")
    print("=" * 50)
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module in modules_to_test:
        if test_module_import(module):
            success_count += 1
    
    print("=" * 50)
    print(f"Résultats: {success_count}/{total_count} modules importés avec succès")
    
    if success_count == total_count:
        print("🎉 Tous les modules sont fonctionnels !")
        return 0
    else:
        print(f"⚠️  {total_count - success_count} modules nécessitent encore des corrections")
        return 1

if __name__ == "__main__":
    sys.exit(main())
