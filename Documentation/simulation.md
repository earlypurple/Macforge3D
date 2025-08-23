# Documentation du Module de Simulation

Ce document décrit les modules de simulation de MacForge3D, qui comprennent l'analyse par éléments finis (FEM) et la simulation thermique.

## Table des matières

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Module FEM](#module-fem)
4. [Module Thermique](#module-thermique)
5. [Exemples d'utilisation](#exemples-dutilisation)
6. [Dépannage](#dépannage)

## Introduction

Les modules de simulation de MacForge3D permettent d'analyser et de valider les modèles 3D avant l'impression. Ils offrent :

- Analyse structurelle par éléments finis
- Simulation thermique pour l'optimisation du refroidissement
- Support pour différents matériaux d'impression 3D
- Recommandations automatiques basées sur les résultats

## Installation

Les modules de simulation nécessitent les dépendances suivantes :

```bash
pip install fenics-dolfin meshio numpy
```

## Module FEM

Le module FEM (`fem_analysis.py`) permet d'analyser les contraintes mécaniques dans les modèles 3D.

### Exemple d'utilisation basique

```python
from simulation.fem_analysis import analyze_model

result = analyze_model(
    mesh_path="model.stl",
    material_name="PLA",
    fixed_points=[(0, 0, 0)],
    forces=[(0, 0, 10, 0, -9.81, 0)]
)
print(result)
```

### Classes principales

#### MaterialProperties

Gère les propriétés mécaniques des matériaux :

- Module de Young (E)
- Coefficient de Poisson (ν)
- Densité
- Limite d'élasticité

#### FEMAnalysis

Effectue l'analyse par éléments finis :

- Chargement du maillage
- Configuration du problème
- Résolution
- Analyse des résultats

### Résultats

L'analyse fournit :

- Contraintes maximales
- Déplacements maximaux
- Facteur de sécurité
- Recommandations

## Module Thermique

Le module thermique (`thermal_sim.py`) simule le comportement thermique pendant l'impression.

### Exemple d'utilisation basique

```python
from simulation.thermal_sim import simulate_thermal

result = simulate_thermal(
    mesh_path="model.stl",
    material_name="PLA",
    initial_temp=200.0,
    ambient_temp=25.0
)
print(result)
```

### Classes principales

#### ThermalProperties

Gère les propriétés thermiques :

- Conductivité thermique
- Capacité thermique spécifique
- Point de fusion
- Température de transition vitreuse

#### ThermalSimulation

Effectue la simulation thermique :

- Chargement du maillage
- Configuration du problème de transfert de chaleur
- Simulation temporelle
- Analyse des résultats

### Résultats

La simulation fournit :

- Températures maximales
- Taux de refroidissement
- Temps passé au-dessus de Tg
- Courbes de température
- Recommandations

## Exemples d'utilisation

### Analyse complète d'un modèle

```python
from simulation.fem_analysis import analyze_model
from simulation.thermal_sim import simulate_thermal

# Analyse structurelle
fem_results = analyze_model(
    "model.stl",
    material_name="PLA",
    fixed_points=[(0, 0, 0)],
    forces=[(0, 0, 10, 0, -9.81, 0)]
)

# Simulation thermique
thermal_results = simulate_thermal(
    "model.stl",
    material_name="PLA",
    initial_temp=200.0,
    ambient_temp=25.0
)

# Analyser les résultats
if fem_results["min_safety_factor"] < 1.5:
    print("Attention: Structure potentiellement faible")
    
if thermal_results["cooling_rate"] > 5:
    print("Attention: Refroidissement trop rapide")
```

### Personnalisation des matériaux

```python
from simulation.fem_analysis import MaterialProperties, FEMAnalysis

# Créer un matériau personnalisé
custom_material = MaterialProperties(
    young_modulus=2.8e9,
    poisson_ratio=0.35,
    density=1100,
    yield_strength=45e6,
    material_name="CUSTOM"
)

# Utiliser dans l'analyse
analysis = FEMAnalysis("model.stl", custom_material)
```

## Dépannage

### Problèmes courants

1. **Erreur de chargement du maillage**
   - Vérifier le format du fichier
   - Vérifier que le maillage est fermé et manifold

2. **Échec de la convergence**
   - Raffiner le maillage
   - Vérifier les conditions aux limites
   - Ajuster les paramètres de simulation

3. **Résultats irréalistes**
   - Vérifier les unités (tout doit être en SI)
   - Vérifier les propriétés des matériaux
   - Vérifier les conditions aux limites

### Optimisation des performances

1. **Réduire le temps de calcul**
   - Simplifier le maillage
   - Réduire le nombre de pas de temps
   - Utiliser la symétrie quand possible

2. **Améliorer la précision**
   - Raffiner le maillage aux zones critiques
   - Augmenter le nombre de pas de temps
   - Utiliser des éléments d'ordre supérieur
