#!/bin/bash

echo "=========================================="
echo "🚀 MacForge3D - Générateur 3D Ultra-Avancé"
echo "=========================================="
echo

# Vérification de Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    echo "💡 Veuillez installer Python3 avec: brew install python3"
    exit 1
fi

echo "✅ Python3 détecté"
echo "🔄 Lancement de MacForge3D..."
echo

# Démarrage du launcher
python3 launcher.py

if [ $? -ne 0 ]; then
    echo
    echo "❌ Erreur lors du lancement"
    echo "💡 Vérifiez que toutes les dépendances sont installées"
    read -p "Appuyez sur Entrée pour continuer..."
fi
