#!/bin/bash
cd "$(dirname "$0")"

clear
echo "🍎 =================================================="
echo "   MacForge3D Professional GUI"
echo "   Interface 3D Complète - Style Blender"
echo "=================================================="
echo

# Vérifications
echo "🔍 Vérification de l'environnement..."

# Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé"
    echo "💡 Installez avec: brew install python3"
    read -p "Appuyez sur Entrée..."
    exit 1
fi

echo "✅ Python3: $(python3 --version)"

# Installation des dépendances si nécessaire
echo "📦 Vérification des dépendances..."

# Liste des dépendances essentielles
dependencies=("numpy" "trimesh" "matplotlib" "pillow" "scipy" "scikit-learn")

missing_deps=()
for dep in "${dependencies[@]}"; do
    python3 -c "import $dep" 2>/dev/null || missing_deps+=("$dep")
done

if [ ${#missing_deps[@]} -gt 0 ]; then
    echo "📦 Installation des dépendances manquantes..."
    for dep in "${missing_deps[@]}"; do
        echo "  📦 Installation de $dep..."
        pip3 install "$dep" --user --quiet
    done
fi

echo "✅ Toutes les dépendances sont disponibles"

# Vérification tkinter
echo "🖥️ Vérification de tkinter..."
python3 -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ tkinter non disponible, tentative d'installation..."
    if command -v brew &> /dev/null; then
        brew install python-tk
    else
        echo "💡 Installez Homebrew et relancez le script"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        read -p "Appuyez sur Entrée..."
        exit 1
    fi
fi

echo "✅ Interface graphique disponible"

echo
echo "🚀 Lancement de MacForge3D Professional GUI..."
echo "   Interface 3D complète avec:"
echo "   • 🎨 Génération IA par texte"
echo "   • 📐 Formes primitives"
echo "   • 🛠️ Outils de modification"
echo "   • 🖥️ Viewport 3D interactif"
echo "   • 📊 Panneaux de propriétés"
echo "   • ⚡ Performance temps réel"
echo

# Lancement de l'interface
python3 MacForge3D_Professional_GUI.py

echo
echo "=================================================="
echo "🎉 Session MacForge3D Professional terminée"
echo "=================================================="

read -p "Appuyez sur Entrée pour fermer..."
