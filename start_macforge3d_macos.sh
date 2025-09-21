#!/bin/bash

# ===================================================================
# 🍎 MacForge3D Launcher pour macOS
# Générateur 3D Ultra-Avancé avec Interface Native macOS
# ===================================================================

clear
echo "🍎 =============================================="
echo "   MacForge3D - Générateur 3D pour macOS"
echo "   Interface Native Optimisée"
echo "=============================================="
echo

# Détection macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Ce script est optimisé pour macOS"
    echo "💡 Utilisez start_macforge3d.sh pour d'autres systèmes"
    echo
fi

# Vérification de Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    echo "💡 Installation recommandée:"
    echo "   • Via Homebrew: brew install python3"
    echo "   • Via site officiel: https://python.org"
    echo
    read -p "Appuyez sur Entrée pour continuer..."
    exit 1
fi

# Vérification de tkinter (interface graphique)
python3 -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  tkinter n'est pas disponible"
    echo "💡 Installation:"
    echo "   brew install python-tk"
    echo
    echo "🔄 Tentative avec le launcher standard..."
    python3 launcher.py
    exit $?
fi

echo "✅ Python3 et tkinter détectés"
echo "🍎 macOS $(sw_vers -productVersion) détecté"
echo "🚀 Lancement de MacForge3D avec interface native..."
echo

# Lancement du launcher macOS optimisé
python3 launcher_macos.py

# Gestion des erreurs
if [ $? -ne 0 ]; then
    echo
    echo "❌ Erreur lors du lancement"
    echo "🔄 Tentative avec le launcher standard..."
    python3 launcher.py
    
    if [ $? -ne 0 ]; then
        echo
        echo "❌ Échec du lancement"
        echo "💡 Vérifications suggérées:"
        echo "   • Installer les dépendances: pip3 install -r requirements.txt"
        echo "   • Vérifier l'installation Python"
        echo "   • Consulter les logs pour plus de détails"
        echo
        read -p "Appuyez sur Entrée pour fermer..."
    fi
fi
