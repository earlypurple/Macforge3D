@echo off
echo ==========================================
echo 🚀 MacForge3D - Générateur 3D Ultra-Avancé
echo ==========================================
echo.

REM Vérification de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé ou pas dans le PATH
    echo 💡 Veuillez installer Python depuis https://python.org
    pause
    exit /b 1
)

echo ✅ Python détecté
echo 🔄 Lancement de MacForge3D...
echo.

REM Démarrage du launcher
python launcher.py

if errorlevel 1 (
    echo.
    echo ❌ Erreur lors du lancement
    echo 💡 Vérifiez que toutes les dépendances sont installées
    pause
)
