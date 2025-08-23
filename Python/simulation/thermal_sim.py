import numpy as np
import meshio
from fenics import *
import logging
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThermalProperties:
    """Classe représentant les propriétés thermiques des matériaux."""
    
    def __init__(
        self,
        thermal_conductivity: float,
        specific_heat: float,
        density: float,
        melting_point: float,
        glass_transition: float,
        material_name: str
    ):
        """
        Initialize les propriétés thermiques du matériau.
        
        Args:
            thermal_conductivity: Conductivité thermique en W/(m·K)
            specific_heat: Capacité thermique spécifique en J/(kg·K)
            density: Densité en kg/m³
            melting_point: Point de fusion en °C
            glass_transition: Température de transition vitreuse en °C
            material_name: Nom du matériau
        """
        self.thermal_conductivity = thermal_conductivity
        self.specific_heat = specific_heat
        self.density = density
        self.melting_point = melting_point
        self.glass_transition = glass_transition
        self.material_name = material_name
        
    @classmethod
    def get_default_materials(cls) -> Dict[str, 'ThermalProperties']:
        """Retourne un dictionnaire des matériaux courants avec leurs propriétés."""
        return {
            'PLA': cls(
                thermal_conductivity=0.13,
                specific_heat=1800,
                density=1240,
                melting_point=180,
                glass_transition=60,
                material_name='PLA'
            ),
            'ABS': cls(
                thermal_conductivity=0.17,
                specific_heat=1400,
                density=1040,
                melting_point=230,
                glass_transition=105,
                material_name='ABS'
            ),
            'PETG': cls(
                thermal_conductivity=0.24,
                specific_heat=1000,
                density=1270,
                melting_point=260,
                glass_transition=80,
                material_name='PETG'
            )
        }

class ThermalSimulation:
    """Classe principale pour la simulation thermique."""
    
    def __init__(
        self,
        mesh_path: str,
        material: ThermalProperties,
        initial_temp: float = 20.0,
        ambient_temp: float = 20.0,
        time_steps: int = 100,
        total_time: float = 1800.0  # 30 minutes par défaut
    ):
        """
        Initialise la simulation thermique.
        
        Args:
            mesh_path: Chemin vers le fichier de maillage
            material: Instance de ThermalProperties
            initial_temp: Température initiale en °C
            ambient_temp: Température ambiante en °C
            time_steps: Nombre de pas de temps
            total_time: Temps total de simulation en secondes
        """
        self.mesh_path = Path(mesh_path)
        self.material = material
        self.initial_temp = initial_temp
        self.ambient_temp = ambient_temp
        self.time_steps = time_steps
        self.total_time = total_time
        
        self.mesh = None
        self.V = None
        self.u = None
        self.u_n = None
        self.results = []
        
    def load_mesh(self) -> bool:
        """Charge et convertit le maillage pour FEniCS."""
        try:
            if self.mesh_path.suffix == '.msh':
                self.mesh = meshio.read(self.mesh_path)
            else:
                temp_msh = self.mesh_path.with_suffix('.msh')
                mesh_in = meshio.read(self.mesh_path)
                meshio.write(temp_msh, mesh_in)
                self.mesh = meshio.read(temp_msh)
            
            logger.info(f"Maillage chargé avec succès: {len(self.mesh.points)} points")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du maillage: {str(e)}")
            return False
            
    def setup_problem(self) -> bool:
        """Configure le problème de transfert de chaleur."""
        try:
            # Créer l'espace des fonctions
            self.V = FunctionSpace(self.mesh, "P", 1)
            
            # Température initiale
            self.u_n = interpolate(Constant(self.initial_temp), self.V)
            
            # Fonction test
            v = TestFunction(self.V)
            
            # Solution au pas de temps suivant
            self.u = Function(self.V)
            
            # Paramètres du matériau
            k = Constant(self.material.thermal_conductivity)
            rho = Constant(self.material.density)
            c = Constant(self.material.specific_heat)
            
            # Pas de temps
            dt = Constant(self.total_time / self.time_steps)
            
            # Forme faible de l'équation de la chaleur
            F = (
                rho * c * (self.u - self.u_n) / dt * v * dx
                + k * inner(grad(self.u), grad(v)) * dx
            )
            
            # Conditions aux limites (convection)
            h = Constant(10.0)  # coefficient de transfert thermique
            T_amb = Constant(self.ambient_temp)
            F += h * (self.u - T_amb) * v * ds
            
            # Définir le problème variationnel
            problem = NonlinearVariationalProblem(F, self.u)
            self.solver = NonlinearVariationalSolver(problem)
            
            logger.info("Problème thermique configuré avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la configuration: {str(e)}")
            return False
            
    def solve(self) -> bool:
        """Résout la simulation thermique dans le temps."""
        try:
            t = 0
            self.results = [(0, self.u_n.vector()[:].copy())]
            
            for n in range(self.time_steps):
                t += self.total_time / self.time_steps
                
                # Résoudre le pas de temps
                self.solver.solve()
                
                # Sauvegarder les résultats
                self.results.append((t, self.u.vector()[:].copy()))
                
                # Mettre à jour la solution
                self.u_n.assign(self.u)
                
                if n % 10 == 0:
                    logger.info(f"t = {t:.1f}s, T_max = {max(self.u.vector()[:]):.1f}°C")
                    
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la résolution: {str(e)}")
            return False
            
    def get_results(self) -> Dict:
        """Analyse les résultats de la simulation."""
        try:
            if not self.results:
                return {"error": "Pas de résultats disponibles"}
                
            # Extraire les températures maximales et minimales
            temps = np.array([r[1] for r in self.results])
            times = np.array([r[0] for r in self.results])
            
            max_temps = np.max(temps, axis=1)
            min_temps = np.min(temps, axis=1)
            
            # Analyser les résultats
            max_temp = float(np.max(max_temps))
            cooling_rate = float(np.mean(np.diff(max_temps) / np.diff(times)))
            time_above_glass = float(np.sum(times[max_temps > self.material.glass_transition]))
            
            results = {
                "max_temperature": max_temp,
                "final_temperature": float(max_temps[-1]),
                "cooling_rate": cooling_rate,  # °C/s
                "time_above_glass_transition": time_above_glass,
                "material": self.material.material_name,
                "temperature_curve": {
                    "times": times.tolist(),
                    "max_temps": max_temps.tolist(),
                    "min_temps": min_temps.tolist()
                }
            }
            
            # Ajouter des recommandations
            results["recommendations"] = []
            
            if max_temp > self.material.melting_point:
                results["recommendations"].append(
                    f"Attention: La température maximale ({max_temp:.1f}°C) dépasse le point de fusion "
                    f"({self.material.melting_point}°C)"
                )
                
            if abs(cooling_rate) > 5:  # °C/s
                results["recommendations"].append(
                    f"Le taux de refroidissement ({abs(cooling_rate):.1f}°C/s) est élevé. "
                    "Risque de déformation."
                )
                
            if time_above_glass > 300:  # 5 minutes
                results["recommendations"].append(
                    f"Temps prolongé ({time_above_glass/60:.1f}min) au-dessus de la température "
                    "de transition vitreuse. Considérez un refroidissement actif."
                )
                
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des résultats: {str(e)}")
            return {"error": str(e)}

def simulate_thermal(
    mesh_path: str,
    material_name: str = "PLA",
    initial_temp: float = 20.0,
    ambient_temp: float = 20.0,
    simulation_time: float = 1800.0,
    time_steps: int = 100
) -> Dict:
    """
    Fonction principale pour lancer une simulation thermique.
    
    Args:
        mesh_path: Chemin vers le fichier de maillage
        material_name: Nom du matériau
        initial_temp: Température initiale en °C
        ambient_temp: Température ambiante en °C
        simulation_time: Durée de simulation en secondes
        time_steps: Nombre de pas de temps
        
    Returns:
        Dict contenant les résultats de la simulation
    """
    try:
        # Obtenir les propriétés du matériau
        materials = ThermalProperties.get_default_materials()
        if material_name not in materials:
            return {"error": f"Matériau '{material_name}' non supporté"}
        material = materials[material_name]
        
        # Créer et configurer la simulation
        sim = ThermalSimulation(
            mesh_path,
            material,
            initial_temp,
            ambient_temp,
            time_steps,
            simulation_time
        )
        
        if not sim.load_mesh():
            return {"error": "Échec du chargement du maillage"}
            
        if not sim.setup_problem():
            return {"error": "Échec de la configuration du problème"}
            
        if not sim.solve():
            return {"error": "Échec de la simulation"}
            
        # Obtenir et retourner les résultats
        return sim.get_results()
        
    except Exception as e:
        logger.error(f"Erreur lors de la simulation: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test avec un modèle simple
    test_model = "path/to/test/model.stl"
    if Path(test_model).exists():
        results = simulate_thermal(
            test_model,
            material_name="PLA",
            initial_temp=200.0,  # Température d'impression typique
            ambient_temp=25.0    # Température ambiante
        )
        print("Résultats de la simulation thermique:")
        print(results)