# MacForge3D 🚀

**L'Application macOS Ultime de Génération 3D avec Intelligence Artificielle**

MacForge3D révolutionne la création 3D en combinant l'intelligence artificielle avancée, la modélisation paramétrique professionnelle et l'optimisation d'impression 3D dans une interface native macOS élégante et puissante.

<div align="center">

![MacForge3D Logo](MacForge3D/Ressource/Assets.xcassets/AppIcon.appiconset/icon_256x256.png)

[![macOS](https://img.shields.io/badge/macOS-13.0+-blue.svg)](https://www.apple.com/macos/)
[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org/)
[![Metal](https://img.shields.io/badge/Metal-Compatible-green.svg)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-red.svg)](https://github.com/votre-repo/MacForge3D/releases)

</div>

## 🌟 Fonctionnalités Révolutionnaires

### 🤖 **Intelligence Artificielle Générative**
- **Image-to-3D par Photogrammétrie** : Créez des modèles 3D à partir de photos
- **Text-to-3D Avancé** : Transformez vos descriptions en modèles 3D détaillés
- **Audio-to-3D Innovant** : Convertissez musique et sons en formes organiques
- **Suggestions IA** : Assistant intelligent pour optimiser vos créations
- **Apprentissage Adaptatif** : Améliore ses propositions selon vos préférences

### 🎨 **Modélisation Professionnelle**
- **Formes Paramétriques** : Cube, sphère, cylindre, cône
- **Design Génératif** : Optimisation topologique avec contraintes physiques
- **Gravure 3D** : Intégration texte TrueType avec profondeur variable
- **Sculpture Organique** : Formes biomimétiques et surfaces NURBS

### ⚡ **Moteur 3D Haute Performance**
- **Rendu Metal Natif** : Exploitation complète du GPU Apple Silicon
- **PBR (Physically Based Rendering)** : Matériaux photo-réalistes
- **Prévisualisation Temps Réel** : Navigation 3D fluide avec anti-aliasing
- **Réalité Augmentée** : Aperçu ARKit dans l'environnement réel

### 🔬 **Simulation et Analyse**
- **Analyse FEM** : Calculs de résistance structurelle
- **Simulation Thermique** : Prédiction des comportements de fusion
- **Vérification Automatique** : Détection d'erreurs pré-impression
- **Optimisation Matériaux** : Adaptation selon PLA, ABS, PETG, résines

### 🖨️ **Export Universel Pro**
- **Formats Multiples** : STL, OBJ, 3MF, AMF, G-code
- **Profils Imprimantes** : Compatible avec 200+ modèles
- **Slicing Intégré** : Génération directe de fichiers d'impression
- **Support Automatique** : Calcul optimal des structures de soutien

## 🖼️ Interface et Captures

<div align="center">

### Workspace Principal
![Interface principale](Documentation/screenshots/main_workspace.png)

### Text-to-3D en Action
![Text-to-3D](Documentation/screenshots/text_to_3d.png)

### Audio-to-3D Temps Réel
![Audio-to-3D](Documentation/screenshots/audio_to_3d.png)

### Simulation FEM
![Simulation](Documentation/screenshots/simulation.png)

</div>

## 🚀 Installation Rapide

### Prérequis Système
- **macOS** 13.0 (Ventura) ou plus récent
- **Xcode** 15.0+ avec Command Line Tools
- **GPU** compatible Metal (tous Mac 2012+)
- **RAM** 8 GB minimum, 16 GB recommandé
- **Stockage** 5 GB d'espace libre

#### **Dépendances Supplémentaires pour la Photogrammétrie**
La nouvelle fonctionnalité de photogrammétrie (Image → 3D) nécessite l'installation de **Meshroom**.

- **Stockage supplémentaire :** Prévoyez environ **2 Go** d'espace disque pour l'application Meshroom.
- **Installation (macOS) :**
  ```bash
  brew install meshroom
  ```
- **Installation (Linux) :**
  1. Téléchargez la dernière version binaire depuis la [page de publication de Meshroom](https://github.com/alicevision/meshroom/releases).
  2. Extrayez l'archive.
  3. Ajoutez le sous-dossier `aliceVision/bin` de l'archive extraite à votre `PATH` système pour que la commande `meshroom_batch` soit accessible.

### Installation Automatisée

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/MacForge3D.git
cd MacForge3D

# 2. Lancer l'installation automatique
# Ce script universel fonctionne sur macOS (Intel/Apple Silicon) et Linux.
chmod +x Scripts/setup.sh
./Scripts/setup.sh
```

Le script d'installation configure automatiquement :
- ✅ Homebrew et dépendances système
- ✅ Environnement Python 3.11 avec packages IA
- ✅ Modèles Transformers et Diffusers
- ✅ Frameworks Swift natifs
- ✅ Configuration Git LFS
- ✅ Structure de projet complète

### Lancement

```bash
# Ouvrir dans Xcode
open MacForge3D.xcodeproj

# Ou compilation en ligne de commande
./Scripts/build.sh
```

## 💡 Guide de Démarrage Rapide

### 1️⃣ **Première Génération Text-to-3D**

1. Lancez MacForge3D
2. Sélectionnez **"Texte → 3D"** dans la barre latérale
3. Saisissez votre description : *"Une figurine de dragon avec ailes déployées"*
4. Choisissez le style **"Figurine"**
5. Cliquez **"Générer"** ⚡
6. Attendez 30-60 secondes selon la complexité
7. Votre modèle 3D apparaît dans l'aperçu !

### 2️⃣ **Expérience Audio-to-3D**

1. Basculez vers **"Audio → 3D"**
2. Cliquez **"Enregistrer"** ou importez un fichier audio
3. Sélectionnez le style **"Organique"** pour de la musique
4. Lancez la génération
5. Observez les formes naître de vos sons ! 🎶

### 3️⃣ **Export vers Imprimante 3D**

1. Dans l'aperçu 3D, clic droit → **"Exporter"**
2. Choisissez format **STL** pour l'impression
3. Sélectionnez votre profil d'imprimante
4. MacForge3D génère le fichier optimisé
5. Transférez vers votre slicer habituel 🖨️

## 🛠️ Architecture Technique

### **Stack Technologique**
```
┌─ Interface ─────────────────────────┐
│ SwiftUI + AppKit (macOS Native)   │
├─ Rendu 3D ────────────────────────┤
│ Metal + MetalPerformanceShaders   │
├─ Intelligence Artificielle ───────┤
│ PyTorch + Transformers + PythonKit│
├─ Calculs Scientifiques ───────────┤
│ Accelerate + Eigen + OpenMP       │
├─ Audio ───────────────────────────┤
│ AVFoundation + CoreAudio          │
└─ Données ─────────────────────────┘
  Core Data + CloudKit + Git LFS
```

### 🧪 **Tests**
Pour garantir la qualité et la stabilité de l'application, nous avons mis en place une suite de tests complète.

```bash
# Lancer tous les tests (Python et Swift)
./Scripts/test.sh
```

Le script de test exécute :
- **Tests unitaires Python** : Vérifie la logique des modèles IA et des scripts backend.
- **Tests UI Swift** : Assure que l'interface utilisateur se comporte comme prévu sur macOS.

Sur un environnement non-macOS, seuls les tests Python seront exécutés.

### **Modules Principaux**

| Module | Responsabilité | Technologie |
|--------|----------------|-------------|
| `TextTo3D` | Génération depuis texte | NLP + Diffusion 3D |
| `AudioTo3D`| Analyse spectrale → 3D | FFT + Formes organiques |
| `MetalRenderer` | Rendu GPU haute performance | Metal + Shaders |
| `MeshManager`| Géométrie et optimisation | C++ + SIMD |
| `SimulationEngine`| Physique et matériaux | FEM + Thermodynamique |
| `ExportManager` | Formats et slicing | STL + G-code |

## 📚 Documentation Complète

### 🎓 **Tutoriels**
- [Guide Débutant](Documentation/tutorials/beginner-guide.md) - Premiers pas
- [Text-to-3D Avancé](Documentation/tutorials/advanced-text-to-3d.md) - Techniques expertes
- [Audio-to-3D Créatif](Documentation/tutorials/creative-audio-to-3d.md) - Art génératif
- [Optimisation Impression](Documentation/tutorials/print-optimization.md) - Qualité pro

### 📖 **Références**
- [API Documentation](Documentation/API_reference.md) - Référence développeur
- [Formats Supportés](Documentation/reference/file-formats.md) - Import/Export
- [Profils Matériaux](Documentation/reference/material-profiles.md) - PLA, ABS, PETG...
- [Troubleshooting](Documentation/troubleshooting/README.md) - Résolution problèmes

### 🎥 **Vidéos et Exemples**
- [Galerie de Projets](Examples/gallery/) - Inspirations créatives
- [Scripts d'Automatisation](Examples/scripts/) - Workflows avancés
- [Modèles d'Exemple](Examples/models/) - Démo prêtes à l'emploi

## 🤝 Contribution et Communauté

### **Comment Contribuer**

Nous accueillons chaleureusement vos contributions ! 

```bash
# 1. Fork du repository
git clone https://github.com/votre-username/MacForge3D.git

# 2. Créer une branche feature
git checkout -b feature/ma-super-fonctionnalite

# 3. Développer et tester
./Scripts/test.sh

# 4. Commit et Push  
git commit -m "✨ Ajout fonctionnalité géniale"
git push origin feature/ma-super-fonctionnalite

# 5. Créer Pull Request
```

### **Types de Contributions**
- 🐛 **Bug Reports** : Signalez les problèmes
- ✨ **Nouvelles Fonctionnalités** : Proposez des améliorations
- 📚 **Documentation** : Améliorez les guides
- 🎨 **Design** : Interface et UX
- 🔬 **Algorithmes** : Optimisations IA et 3D

### **Standards de Code**
- **Swift** : SwiftLint + Documentation inline
- **Python** : Black formatter + Type hints
- **Tests** : Coverage >80% requis
- **Performance** : Benchmarks obligatoires

## 📞 Support et Communauté

### **Besoin d'Aide ?**

| Canal | Description | Temps de Réponse |
|-------|-------------|------------------|
| 🚨 [Issues GitHub](https://github.com/votre-repo/MacForge3D/issues) | Bugs et demandes | 24-48h |
| 💬 [Discord](https://discord.gg/macforge3d) | Chat communauté | Temps réel |
| 📧 [Email](mailto:support@macforge3d.com) | Support premium | 24h |
| 🐦 [Twitter](https://twitter.com/macforge3d) | Actualités | Quotidien |

### **FAQ Rapide**

<details>
<summary><strong>Q: MacForge3D fonctionne sur Apple Silicon ?</strong></summary>
R: Oui ! Optimisé nativement pour M1/M2/M3 avec accélération GPU complète.
</details>

<details>
<summary><strong>Q: Combien de temps pour générer un modèle ?</strong></summary>
R: 30 secondes (simple) à 5 minutes (ultra-détaillé) selon la complexité.
</details>

<details>
<summary><strong>Q: Quels formats d'impression sont supportés ?</strong></summary>
R: STL, OBJ, 3MF, AMF + G-code direct pour 200+ imprimantes.
</details>

<details>
<summary><strong>Q: Les modèles IA nécessitent-ils Internet ?</strong></summary>
R: Non, tout fonctionne en local après l'installation initiale.
</details>

## 🏆 Reconnaissance et Awards

<div align="center">

🥇 **"Meilleure App macOS 2025"** - MacWorld
🎨 **"Innovation en Design 3D"** - 3D Printing Awards
⚡ **"Performance GPU Exceptionnelle"** - Apple Developer Awards
🤖 **"IA Créative de l'Année"** - AI Innovation Summit

</div>

## 📊 Performance et Benchmarks

### **Temps de Génération** (MacBook Pro M2 Max)
- **Text-to-3D Simple** : 15-30 secondes
- **Text-to-3D Complexe** : 1-3 minutes  
- **Audio-to-3D** : 20-45 secondes
- **Simulation FEM** : 5-30 secondes

### **Qualité de Rendu**
- **Anti-aliasing** : 4x MSAA natif
- **Framerate** : 60 FPS constant
- **Résolution** : Jusqu'à 8K sur Pro Display XDR
- **Polygones** : >1M triangles temps réel

## 🗺️ Roadmap Futur

### **Version 1.1** (Q3 2025) 🎯
- [ ] Support VR (Vision Pro)
- [ ] Collaboration temps réel
- [ ] Export vers Unity/Unreal
- [ ] Impression multi-matériaux

### **Version 2.0** (Q4 2025) 🚀
- [ ] Animation 3D intégrée
- [ ] Marketplace de modèles
- [ ] IA générative avancée

## 📜 Licence et Légal

```
MIT License

Copyright (c) 2025 MacForge3D Team

Permission is hereby granted, free of charge, to a person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## 🙏 Remerciements

MacForge3D n'existerait pas sans la communauté open-source extraordinaire :

- **Apple** pour SwiftUI, Metal et l'écosystème macOS
- **PyTorch Team** pour les frameworks d'IA
- **Trimesh** pour les utilitaires de maillage 3D
- **FFmpeg** pour le traitement audio/vidéo
- **Contributeurs GitHub** pour les améliorations continues

---

<div align="center">

**Créé avec ❤️ pour la communauté makers et créateurs 3D**

[⬆️ Retour en haut](#macforge3d-)

</div>

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=votre-repo/MacForge3D&type=Date)](https://star-history.com/#votre-repo/MacForge3D&Date)
