# Guide d'Intégration Bambu Lab pour MacForge3D

## 🖨️ Vue d'ensemble

MacForge3D propose maintenant une intégration complète avec toute la gamme d'imprimantes Bambu Lab, incluant le support avancé AMS, la détection IA, et l'optimisation spécifique pour chaque modèle d'imprimante.

## 📋 Imprimantes Supportées

### Gamme Complète Bambu Lab

| Modèle | Volume (mm) | AMS | LiDAR | IA | Chambre | Vitesse Max |
|--------|-------------|-----|-------|----|---------| ------------|
| **A1 mini** | 180×180×180 | ✅ 4 slots | ✅ | ✅ | ❌ | 500 mm/s |
| **A1** | 256×256×256 | ✅ 4 slots | ✅ | ✅ | ❌ | 500 mm/s |
| **P1P** | 256×256×256 | ✅ 4 slots | ✅ | ✅ | ❌ | 500 mm/s |
| **P1S** | 256×256×256 | ✅ 4 slots | ✅ | ✅ | ✅ | 500 mm/s |
| **X1** | 256×256×256 | ✅ 4 slots | ✅ | ✅ | ✅ | 500 mm/s |
| **X1 Carbon** | 256×256×256 | ✅ 4 slots | ✅ | ✅ | ✅ | 500 mm/s |

## 🎯 Fonctionnalités Principales

### 🤖 Support AMS (Automatic Material System)
- **Gestion automatique des 4 slots**
- **Détection automatique du type de filament**
- **Changement automatique de matériau en cours d'impression**
- **Optimisation des tours de purge**
- **Support multi-matériaux et multi-couleurs**

### 👁️ LiDAR et Détection IA
- **Inspection de première couche automatique**
- **Détection d'erreurs en temps réel**
- **Surveillance de la qualité d'impression**
- **Ajustements automatiques basés sur l'IA**

### 🔧 Paramètres d'Impression Optimisés
- **Profils matériaux spécifiques Bambu Lab**
- **Hauteur de couche adaptative**
- **Optimisation de vitesse intelligente**
- **Gestion de la pression d'avance**
- **Calibration automatique du flux**

## 🚀 Guide d'Utilisation Rapide

### 1. Configuration de Base

```swift
// Créer une configuration pour X1 Carbon
let config = BambuPrinterConfig(
    model: .X1_CARBON,
    serial_number: "AC001234567890",
    ip_address: "192.168.1.100",
    access_code: "12345678"
)
```

### 2. Paramètres d'Impression

```swift
// Paramètres optimisés pour PLA sur X1 Carbon
var settings = BambuPrintSettings()
settings.nozzle_temperature = 220
settings.bed_temperature = 65
settings.print_speed = 300
settings.ams_enabled = true
settings.active_ams_slot = 1
settings.ai_monitoring = true
settings.lidar_calibration = true
```

### 3. Export Optimisé

```swift
let bambuExporter = BambuLabExporter()
let exportOptions = BambuExportOptions(
    printerProfile: profile,
    useAMS: true,
    enableAIInspection: true,
    enableLidar: true
)

try bambuExporter.exportForBambuLab(
    model: model3D,
    options: exportOptions,
    outputURL: outputURL
)
```

## 🎨 Intégration Python

### Configuration Bambu Lab

```python
from modern_tech.bambu_lab_integration import (
    create_bambu_config,
    BambuLabIntegration,
    BambuPrintSettings
)

# Créer la configuration
config = create_bambu_config(
    model="X1 Carbon",
    serial_number="AC001234567890",
    ip_address="192.168.1.100",
    access_code="12345678"
)

# Initialiser l'intégration
bambu = BambuLabIntegration(config)
```

### Lancement d'Impression

```python
import asyncio
from pathlib import Path

async def print_model():
    # Paramètres recommandés pour ABS
    settings = bambu.get_recommended_settings("ABS", "fine")
    
    # Configurer l'AMS
    settings.ams_enabled = True
    settings.active_ams_slot = 2  # Slot avec ABS
    
    # Lancer l'impression
    stl_file = Path("model.stl")
    success = await bambu.print_model(stl_file, settings)
    
    if success:
        print("🎉 Impression lancée avec succès!")
    else:
        print("❌ Erreur lors du lancement")

# Exécuter
asyncio.run(print_model())
```

### API Locale (WebSocket)

```python
# Connexion directe à l'imprimante
api = BambuLocalAPI(config)
await api.connect()

# Récupérer le statut
status = await api.get_status()
print(f"État: {status['state']}")

# Statut AMS détaillé
ams_slots = await api.get_ams_status()
for slot in ams_slots:
    print(f"Slot {slot.slot_id}: {slot.material_type} ({slot.status.value})")

# Calibration LiDAR
if config.has_lidar:
    success = await api.calibrate_lidar()
    print(f"Calibration LiDAR: {'✅' if success else '❌'}")
```

## 🛠️ Profils Matériaux Avancés

### PLA Optimisé Bambu Lab
```python
pla_profile = BambuMaterialProfile(
    name="PLA Basic",
    nozzle_temp=220,
    bed_temp=65,
    chamber_temp=None,
    print_speed=500,  # Vitesse max Bambu Lab
    retraction=0.8,
    pressure_advance=0.02,
    flow_rate=0.98
)
```

### ABS Haute Performance
```python
abs_profile = BambuMaterialProfile(
    name="ABS High Temp",
    nozzle_temp=260,
    bed_temp=90,
    chamber_temp=40,  # Pour modèles avec chambre
    print_speed=300,
    retraction=0.5,
    pressure_advance=0.025,
    flow_rate=0.96
)
```

### TPU Flexible
```python
tpu_profile = BambuMaterialProfile(
    name="TPU 95A",
    nozzle_temp=230,
    bed_temp=50,
    chamber_temp=None,
    print_speed=30,   # Vitesse réduite pour TPU
    retraction=0.0,   # Pas de rétraction
    pressure_advance=0.0,
    flow_rate=1.05
)
```

## 📊 Génération G-code Optimisé

Le moteur de slicing génère du G-code spécifiquement optimisé pour Bambu Lab:

### En-tête Bambu Lab
```gcode
; Generated by MacForge3D for Bambu Lab
; Printer: X1 Carbon
; Nozzle diameter: 0.4mm
; AMS enabled with 4 slots
; AMS Slot 1: PLA
; AMS Slot 2: ABS
; Lidar first layer inspection: enabled
; AI failure detection: enabled

M981 S1  ; Enable AI monitoring
M971 S1  ; Enable LiDAR inspection
```

### Configuration AMS
```gcode
M620 S1 A  ; Load filament from AMS slot 1
G92 E0     ; Reset extruder
M621 S1    ; Select AMS slot 1
```

### Optimisations Spécifiques
```gcode
M106 S128  ; Fan speed pour Bambu Lab
M207 S0.8  ; Rétraction optimisée
M572 S0.02 ; Pressure advance
```

## 🎯 Formats d'Export

### .3mf Optimisé Bambu Studio
```python
# Générateur .3mf avec métadonnées Bambu Lab
generator = ThreeMFGenerator(config)

success = generator.create_3mf_file(
    stl_file=Path("model.stl"),
    settings=settings,
    output_file=Path("model_bambu.3mf")
)

# Le fichier .3mf inclut:
# - Métadonnées Bambu Studio
# - Configuration AMS
# - Paramètres d'impression
# - Profils matériaux
```

### G-code Direct
```python
# Génération G-code directe
slice_engine = BambuSliceEngine(config)
gcode = slice_engine.generate_optimized_gcode(
    stl_file=Path("model.stl"),
    settings=settings
)

# Le G-code inclut:
# - Commandes spécifiques Bambu Lab
# - Configuration AMS automatique
# - Surveillance IA activée
# - Calibration LiDAR
```

## 🔧 Dépannage

### Problèmes Courants

**❌ Connexion WebSocket échouée**
```python
# Vérifier l'adresse IP et le code d'accès
config.ip_address = "192.168.1.100"  # IP correcte
config.access_code = "12345678"       # Code de l'imprimante

# Test de connexion
try:
    await api.connect()
    print("✅ Connexion réussie")
except Exception as e:
    print(f"❌ Erreur: {e}")
```

**❌ AMS non détecté**
```python
# Vérifier la configuration AMS
if not config.has_ams:
    print("⚠️ Cette imprimante n'a pas d'AMS")
else:
    # Vérifier les slots AMS
    ams_slots = await api.get_ams_status()
    for slot in ams_slots:
        if slot.status == AMSSlotStatus.EMPTY:
            print(f"⚠️ Slot {slot.slot_id} est vide")
```

**❌ LiDAR non fonctionnel**
```python
# Vérifier le support LiDAR
if not config.has_lidar:
    print("⚠️ Cette imprimante n'a pas de LiDAR")
else:
    # Test calibration
    success = await api.calibrate_lidar()
    if not success:
        print("❌ Échec calibration LiDAR")
```

### Optimisations Performance

**🚀 Cache des paramètres**
```python
# Utiliser le cache pour les paramètres fréquents
from modern_tech.smart_cache import cache_result, get_cached_result

# Mettre en cache les paramètres optimisés
cache_key = f"bambu_settings_{material}_{quality}"
cached_settings = await get_cached_result("bambu_settings", cache_key)

if not cached_settings:
    settings = bambu.get_recommended_settings(material, quality)
    await cache_result("bambu_settings", cache_key, settings, ttl=3600)
```

**⚡ WebAssembly pour gros modèles**
```python
# Utiliser WebAssembly pour l'optimisation
from modern_tech.webassembly_bridge import wasm_optimize_mesh

# Optimiser le maillage avant export
vertices = model.get_vertices()
faces = model.get_faces()

optimized = await wasm_optimize_mesh(vertices, faces, reduction=0.1)
if optimized.get("success"):
    print(f"🚀 Maillage optimisé: {optimized['speedup_factor']}x plus rapide")
```

## 📚 Ressources Supplémentaires

- **[Documentation Bambu Lab](https://wiki.bambulab.com/)**
- **[API WebSocket Bambu Lab](https://github.com/bambulab/BambuStudio/wiki)**
- **[Profils Matériaux](https://github.com/bambulab/BambuStudio/tree/master/resources/profiles)**
- **[G-code Bambu Lab](https://wiki.bambulab.com/en/software/bambu-studio/gcode)**

## 🎉 Conclusion

L'intégration Bambu Lab dans MacForge3D offre:

- **✅ Support complet de tous les modèles Bambu Lab**
- **🤖 Intégration AMS avancée**
- **👁️ Surveillance IA et LiDAR**
- **⚡ Performance optimisée**
- **🎯 G-code spécifiquement optimisé**
- **📱 Contrôle en temps réel**

Cette intégration place MacForge3D à la pointe de la technologie d'impression 3D avec Bambu Lab!