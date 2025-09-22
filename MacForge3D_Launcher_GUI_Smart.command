#!/bin/bash

cd "$(dirname "$0")"

clear
echo "🍎 ========================================================"
echo "   MacForge3D Ultra Performance GUI Edition"
echo "   Interface 3D Professionnelle pour macOS"
echo "   🎨 Comme Blender • ⚡ Performance SolidWorks"
echo "========================================================"
echo

# Vérifications système
echo "🔍 Vérifications système..."

# Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé"
    echo "💡 Installez avec: brew install python3"
    read -p "Appuyez sur Entrée pour fermer..."
    exit 1
fi

echo "✅ Python3: $(python3 --version)"

echo
echo "🚀 Lancement MacForge3D GUI..."
echo

# Essayer d'abord la version complète
echo "🎨 Tentative de lancement GUI complet..."
python3 Python/macforge3d_gui.py 2>/dev/null

# Si échec, vérifier les dépendances
if [ $? -ne 0 ]; then
    echo
    echo "⚠️ Échec du lancement GUI complet"
    echo "🔍 Vérification des dépendances..."
    
    # Tester numpy
    python3 -c "import numpy" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "❌ NumPy manquant"
        echo
        echo "🔧 SOLUTIONS DISPONIBLES:"
        echo "   1. Lancer Fix_Dependencies_GUI.command"
        echo "   2. Installer manuellement: pip3 install numpy matplotlib"
        echo "   3. Utiliser la version simple (sans dépendances)"
        echo
        
        read -p "Voulez-vous lancer la version simple ? (y/N): " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo
            echo "🎨 Lancement MacForge3D Simple GUI..."
            python3 Python/macforge3d_simple_gui.py
        else
            echo
            echo "💡 Pour installer les dépendances automatiquement:"
            echo "   Double-cliquez sur Fix_Dependencies_GUI.command"
        fi
    else
        echo "❌ Erreur inconnue lors du lancement"
        echo "💡 Essayez de relancer ou utilisez Fix_Dependencies_GUI.command"
    fi
fi

echo
echo "========================================================"
echo "🏁 MacForge3D GUI fermé"
echo "========================================================"

read -p "Appuyez sur Entrée pour fermer..."
