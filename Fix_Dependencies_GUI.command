#!/bin/bash

cd "$(dirname "$0")"

clear
echo "🔧 ========================================================"
echo "   MacForge3D GUI - Correction Automatique"
echo "   Installation des dépendances manquantes"
echo "========================================================"
echo

echo "🔍 Diagnostic du problème..."
echo "❌ Erreur détectée: ModuleNotFoundError: No module named 'numpy'"
echo

# Vérifier quelle version de Python est utilisée
echo "🐍 Vérification de Python..."
echo "Python system: $(which python3)"
echo "Version: $(python3 --version)"

# Vérifier pip3
echo
echo "📦 Vérification de pip3..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 non trouvé, installation..."
    python3 -m ensurepip --upgrade
else
    echo "✅ pip3 trouvé: $(which pip3)"
fi

# Mise à jour pip
echo
echo "🔄 Mise à jour de pip..."
python3 -m pip install --upgrade pip

# Installation forcée des dépendances essentielles
echo
echo "⚡ Installation FORCÉE des dépendances essentielles..."

dependencies=("numpy" "matplotlib" "scipy" "pillow")

for dep in "${dependencies[@]}"; do
    echo
    echo "📦 Installation FORCÉE de $dep..."
    
    # Essayer plusieurs méthodes d'installation
    echo "  Méthode 1: pip3 install..."
    pip3 install "$dep" --user --upgrade --force-reinstall || echo "  ❌ Méthode 1 échouée"
    
    echo "  Méthode 2: python3 -m pip install..."
    python3 -m pip install "$dep" --user --upgrade --force-reinstall || echo "  ❌ Méthode 2 échouée"
    
    echo "  Méthode 3: Installation système..."
    sudo pip3 install "$dep" --upgrade --force-reinstall 2>/dev/null || echo "  ❌ Méthode 3 échouée (normal)"
    
    # Vérification
    python3 -c "import $dep; print(f'✅ $dep installé avec succès')" 2>/dev/null || echo "  ⚠️ $dep toujours problématique"
done

# Vérification spéciale pour tkinter
echo
echo "🖥️ Vérification tkinter..."
python3 -c "import tkinter; print('✅ tkinter disponible')" 2>/dev/null || {
    echo "❌ tkinter manquant, installation via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install python-tk
    else
        echo "⚠️ Homebrew non trouvé, tkinter pourrait être manquant"
    fi
}

# Test complet des imports
echo
echo "🧪 Test complet des dépendances..."
python3 -c "
import sys
print(f'Python: {sys.version}')

modules = ['numpy', 'matplotlib', 'tkinter', 'scipy', 'PIL']
missing = []

for module in modules:
    try:
        __import__(module)
        print(f'✅ {module} - OK')
    except ImportError as e:
        print(f'❌ {module} - MANQUANT: {e}')
        missing.append(module)

if missing:
    print(f'⚠️ Modules manquants: {missing}')
    print('💡 Essayez de les installer manuellement avec:')
    for mod in missing:
        print(f'   pip3 install {mod} --user')
else:
    print('🎉 Toutes les dépendances sont prêtes!')
"

echo
echo "========================================================"
echo "🚀 Relancement de MacForge3D GUI..."
echo "========================================================"
echo

# Relancer l'application
python3 Python/macforge3d_gui.py

echo
echo "========================================================"
echo "🔧 Script de correction terminé"
echo "========================================================"

read -p "Appuyez sur Entrée pour fermer..."
