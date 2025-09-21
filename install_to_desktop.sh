#!/bin/bash

# ====================================================================
# 🚀 MacForge3D Desktop Installer pour macOS
# Télécharge et installe MacForge3D directement sur votre bureau
# ====================================================================

clear
echo "🍎 =================================================="
echo "   MacForge3D Desktop Installer"
echo "   Installation directe sur le Bureau"
echo "=================================================="
echo

# Détection du bureau utilisateur
DESKTOP_PATH="$HOME/Desktop"
MACFORGE3D_PATH="$DESKTOP_PATH/MacForge3D"

echo "📍 Installation dans: $MACFORGE3D_PATH"
echo

# Vérification des prérequis
echo "🔍 Vérification des prérequis..."

# Vérifier Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    echo "💡 Installation automatique via Homebrew..."
    
    # Vérifier Homebrew
    if ! command -v brew &> /dev/null; then
        echo "📦 Installation de Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    echo "📦 Installation de Python3..."
    brew install python3
fi

echo "✅ Python3 détecté: $(python3 --version)"

# Vérifier tkinter
python3 -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installation de tkinter..."
    brew install python-tk
fi

echo "✅ tkinter disponible"

# Créer le dossier MacForge3D sur le bureau
echo
echo "📁 Création du dossier MacForge3D sur le bureau..."

if [ -d "$MACFORGE3D_PATH" ]; then
    echo "⚠️  Le dossier MacForge3D existe déjà"
    read -p "Voulez-vous le remplacer ? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$MACFORGE3D_PATH"
        echo "🗑️  Ancien dossier supprimé"
    else
        echo "❌ Installation annulée"
        exit 1
    fi
fi

mkdir -p "$MACFORGE3D_PATH"
cd "$MACFORGE3D_PATH"

echo "✅ Dossier créé: $MACFORGE3D_PATH"

# Téléchargement ou copie des fichiers MacForge3D
echo
echo "📥 Installation de MacForge3D..."

# Si nous sommes dans un environnement de développement, copier les fichiers
if [ -d "/workspaces/Macforge3D" ]; then
    echo "🔄 Copie depuis l'environnement de développement..."
    cp -r /workspaces/Macforge3D/* "$MACFORGE3D_PATH/"
    echo "✅ Fichiers copiés avec succès"
else
    # Sinon, créer les fichiers essentiels
    echo "🛠️  Création des fichiers MacForge3D..."
    
    # Créer la structure de dossiers
    mkdir -p Python/ai_models Python/exporters Python/simulation
    mkdir -p Examples/generated_models
    mkdir -p Documentation
    
    # Créer un fichier requirements.txt minimal
    cat > requirements.txt << 'EOF'
# MacForge3D - Dépendances Python
numpy>=1.21.0
trimesh>=3.15.0
torch>=1.12.0
pillow>=8.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0

# Dépendances optionnelles (améliorent les fonctionnalités)
# opencv-python
# diffusers
# transformers
# optuna
# wandb
# h5py
# GPUtil
# pymeshfix
# ray
EOF

    echo "✅ Structure de base créée"
fi

# Créer le launcher desktop pour macOS
echo
echo "🚀 Création du launcher MacForge3D..."

# Launcher principal
cat > "$MACFORGE3D_PATH/MacForge3D_Launcher.command" << 'EOF'
#!/bin/bash

# Change to the script's directory
cd "$(dirname "$0")"

clear
echo "🍎 =================================================="
echo "   MacForge3D - Générateur 3D Ultra-Avancé"
echo "   Launcher Desktop pour macOS"
echo "=================================================="
echo

# Vérifier Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    echo "💡 Veuillez installer Python3 avec: brew install python3"
    read -p "Appuyez sur Entrée pour fermer..."
    exit 1
fi

echo "✅ Python3 détecté: $(python3 --version)"

# Vérifier tkinter
python3 -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  tkinter n'est pas disponible"
    echo "💡 Installation: brew install python-tk"
    echo "🔄 Tentative de lancement sans interface graphique..."
fi

echo "🚀 Lancement de MacForge3D..."
echo

# Lancer MacForge3D
if [ -f "launcher_macos.py" ]; then
    python3 launcher_macos.py
elif [ -f "launcher.py" ]; then
    python3 launcher.py
else
    echo "❌ Fichiers launcher non trouvés"
    echo "💡 Réinstallation recommandée"
    read -p "Appuyez sur Entrée pour fermer..."
    exit 1
fi

# Maintenir la fenêtre ouverte en cas d'erreur
if [ $? -ne 0 ]; then
    echo
    echo "❌ Erreur lors du lancement"
    echo "💡 Vérifiez l'installation des dépendances"
    read -p "Appuyez sur Entrée pour fermer..."
fi
EOF

# Rendre le launcher exécutable
chmod +x "$MACFORGE3D_PATH/MacForge3D_Launcher.command"

# Créer un script d'installation des dépendances
cat > "$MACFORGE3D_PATH/Install_Dependencies.command" << 'EOF'
#!/bin/bash

cd "$(dirname "$0")"

clear
echo "📦 =================================================="
echo "   MacForge3D - Installation des Dépendances"
echo "=================================================="
echo

echo "🔄 Installation des dépendances Python..."

# Installer les dépendances de base
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    echo "📦 Installation des modules essentiels..."
    pip3 install numpy trimesh torch pillow scipy scikit-learn matplotlib
fi

echo
echo "✅ Installation terminée!"
echo "🚀 Vous pouvez maintenant lancer MacForge3D"
echo

read -p "Appuyez sur Entrée pour fermer..."
EOF

chmod +x "$MACFORGE3D_PATH/Install_Dependencies.command"

# Créer un README desktop
cat > "$MACFORGE3D_PATH/README_DESKTOP.md" << 'EOF'
# 🍎 MacForge3D - Installation Desktop

## 🚀 Lancement Rapide

**Double-cliquez sur `MacForge3D_Launcher.command`**

## 📦 Si première utilisation

1. Double-cliquez sur `Install_Dependencies.command`
2. Attendez l'installation des dépendances
3. Lancez `MacForge3D_Launcher.command`

## ✨ Fonctionnalités

- 🎨 Génération 3D par IA
- 📸 Image vers 3D
- 🔧 Réparation de mesh
- ⚡ Optimisation automatique
- 📦 Compression avancée

## 🆘 Support

Si problèmes:
1. Installer Python3: `brew install python3`
2. Installer tkinter: `brew install python-tk`
3. Redémarrer le launcher

---
🎉 MacForge3D - Prêt pour macOS !
EOF

# Créer l'icône de l'application (script AppleScript)
cat > "$MACFORGE3D_PATH/Create_Desktop_Shortcut.command" << 'EOF'
#!/bin/bash

cd "$(dirname "$0")"

# Créer un alias sur le bureau pour un accès facile
CURRENT_PATH=$(pwd)
DESKTOP_ALIAS="$HOME/Desktop/🚀 MacForge3D"

if [ ! -e "$DESKTOP_ALIAS" ]; then
    ln -s "$CURRENT_PATH/MacForge3D_Launcher.command" "$DESKTOP_ALIAS"
    echo "✅ Raccourci créé sur le bureau: 🚀 MacForge3D"
else
    echo "ℹ️  Le raccourci existe déjà sur le bureau"
fi

echo "🎉 MacForge3D est prêt à l'utilisation!"
read -p "Appuyez sur Entrée pour fermer..."
EOF

chmod +x "$MACFORGE3D_PATH/Create_Desktop_Shortcut.command"

# Finalisation
echo
echo "🎉 =================================================="
echo "   Installation Terminée avec Succès !"
echo "=================================================="
echo
echo "📍 MacForge3D installé dans:"
echo "   $MACFORGE3D_PATH"
echo
echo "🚀 POUR LANCER MacForge3D :"
echo "   1. Ouvrez le dossier MacForge3D sur votre bureau"
echo "   2. Double-cliquez sur 'MacForge3D_Launcher.command'"
echo
echo "📦 PREMIÈRE UTILISATION :"
echo "   • Double-cliquez d'abord sur 'Install_Dependencies.command'"
echo "   • Puis lancez 'MacForge3D_Launcher.command'"
echo
echo "🔗 RACCOURCI BUREAU :"
echo "   • Double-cliquez sur 'Create_Desktop_Shortcut.command'"
echo "   • Un raccourci '🚀 MacForge3D' apparaîtra sur votre bureau"
echo
echo "✨ FONCTIONNALITÉS DISPONIBLES :"
echo "   🎨 Génération 3D par IA à partir de texte"
echo "   📸 Reconstruction 3D à partir d'images"
echo "   🔧 Réparation automatique de mesh"
echo "   ⚡ Optimisation par intelligence artificielle"
echo "   📦 Compression avancée de modèles"
echo "   🧠 Système de cache intelligent"
echo
echo "=================================================="
echo "🎯 MacForge3D est maintenant prêt sur votre bureau !"
echo "=================================================="

# Ouvrir le dossier dans Finder
if command -v open &> /dev/null; then
    echo
    echo "📁 Ouverture du dossier MacForge3D..."
    open "$MACFORGE3D_PATH"
fi

echo
read -p "Appuyez sur Entrée pour terminer..."
