import numpy as np
import meshio
from fenics import *
import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaterialProperties:
    """Classe représentant les propriétés des matériaux pour l'analyse FEM."""
    
    def __init__(
        self,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
        yield_strength: float,
        material_name: str
    ):
        """
        Initialize les propriétés du matériau.
        
        Args:
            young_modulus: Module de Young (E) en Pa
            poisson_ratio: Coefficient de Poisson (ν)
            density: Densité en kg/m³
            yield_strength: Limite d'élasticité en Pa
            material_name: Nom du matériau (ex: "PLA", "ABS", etc.)
        """
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.density = density
        self.yield_strength = yield_strength
        self.material_name = material_name
        
    @classmethod
    def get_default_materials(cls) -> Dict[str, 'MaterialProperties']:
        """Retourne un dictionnaire des matériaux courants avec leurs propriétés."""
        return {
            'PLA': cls(
                young_modulus=3.5e9,  # 3.5 GPa
                poisson_ratio=0.36,
                density=1240,
                yield_strength=50e6,  # 50 MPa
                material_name='PLA'
            ),
            'ABS': cls(
                young_modulus=2.3e9,  # 2.3 GPa
                poisson_ratio=0.35,
                density=1040,
                yield_strength=40e6,  # 40 MPa
                material_name='ABS'
            ),
            'PETG': cls(
                young_modulus=2.1e9,  # 2.1 GPa
                poisson_ratio=0.38,
                density=1270,
                yield_strength=50e6,  # 50 MPa
                material_name='PETG'
            )
        }

class FEMAnalysis:
    """Classe principale pour l'analyse par éléments finis."""
    
    def __init__(self, mesh_path: str, material: MaterialProperties):
        """
        Initialise l'analyse FEM.
        
        Args:
            mesh_path: Chemin vers le fichier de maillage (.msh, .stl, etc.)
            material: Instance de MaterialProperties pour le matériau à analyser
        """
        self.mesh_path = Path(mesh_path)
        self.material = material
        self.mesh = None
        self.V = None
        self.solution = None
        self.stress_field = None
        
    def load_mesh(self) -> bool:
        """Charge et convertit le maillage pour FEniCS."""
        try:
            if self.mesh_path.suffix == '.msh':
                self.mesh = meshio.read(self.mesh_path)
            else:
                # Convertir d'autres formats en .msh si nécessaire
                temp_msh = self.mesh_path.with_suffix('.msh')
                mesh_in = meshio.read(self.mesh_path)
                meshio.write(temp_msh, mesh_in)
                self.mesh = meshio.read(temp_msh)
            
            logger.info(f"Maillage chargé avec succès: {len(self.mesh.points)} points")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du maillage: {str(e)}")
            return False
            
    def setup_problem(
        self,
        fixed_boundaries: List[Tuple[float, float, float]],
        forces: List[Tuple[float, float, float, float, float, float]]
    ) -> bool:
        """
        Configure le problème FEM avec les conditions aux limites.
        
        Args:
            fixed_boundaries: Liste des points où le déplacement est fixé [(x,y,z), ...]
            forces: Liste des forces appliquées [(x,y,z, Fx,Fy,Fz), ...]
        """
        try:
            # Créer l'espace des fonctions
            self.V = VectorFunctionSpace(self.mesh, "P", 1)
            
            # Définir les conditions aux limites
            u = TrialFunction(self.V)
            v = TestFunction(self.V)
            
            # Forme bilinéaire (élasticité linéaire)
            E = self.material.young_modulus
            nu = self.material.poisson_ratio
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            
            def epsilon(u):
                return 0.5 * (grad(u) + grad(u).T)
                
            def sigma(u):
                return lmbda * tr(epsilon(u)) * Identity(3) + 2.0 * mu * epsilon(u)
                
            a = inner(sigma(u), epsilon(v)) * dx
            
            # Forme linéaire (forces)
            L = Constant((0.0, 0.0, 0.0))
            for force in forces:
                f = Constant((force[3], force[4], force[5]))
                L = L + dot(f, v) * ds
                
            # Assembler le système
            A = assemble(a)
            b = assemble(L)
            
            logger.info("Problème FEM configuré avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la configuration du problème: {str(e)}")
            return False
            
    def solve(self) -> bool:
        """Résout le problème FEM."""
        try:
            # Résoudre le système
            solve(A, self.solution.vector(), b)
            
            # Calculer le champ de contraintes
            self.stress_field = project(sigma(self.solution), TensorFunctionSpace(self.mesh, "P", 1))
            
            logger.info("Analyse FEM terminée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la résolution: {str(e)}")
            return False
            
    def get_results(self) -> Dict:
        """Retourne les résultats de l'analyse."""
        if not self.solution or not self.stress_field:
            return {"error": "Aucune solution disponible"}
            
        try:
            # Calculer les contraintes de von Mises
            von_mises = sqrt(
                0.5 * (
                    (self.stress_field[0,0] - self.stress_field[1,1])**2 +
                    (self.stress_field[1,1] - self.stress_field[2,2])**2 +
                    (self.stress_field[2,2] - self.stress_field[0,0])**2 +
                    6.0 * (
                        self.stress_field[0,1]**2 +
                        self.stress_field[1,2]**2 +
                        self.stress_field[2,0]**2
                    )
                )
            )
            
            # Calculer le facteur de sécurité
            safety_factor = self.material.yield_strength / von_mises
            
            # Obtenir les déplacements maximaux
            u_magnitude = sqrt(
                self.solution[0]**2 +
                self.solution[1]**2 +
                self.solution[2]**2
            )
            max_displacement = max(u_magnitude.vector()[:])
            
            return {
                "max_stress": float(max(von_mises.vector()[:])),
                "max_displacement": float(max_displacement),
                "min_safety_factor": float(min(safety_factor.vector()[:])),
                "material": self.material.material_name
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des résultats: {str(e)}")
            return {"error": str(e)}

def analyze_model(
    mesh_path: str,
    material_name: str = "PLA",
    fixed_points: Optional[List[Tuple[float, float, float]]] = None,
    forces: Optional[List[Tuple[float, float, float, float, float, float]]] = None
) -> Dict:
    """
    Fonction principale pour analyser un modèle 3D.
    
    Args:
        mesh_path: Chemin vers le fichier de maillage
        material_name: Nom du matériau à utiliser
        fixed_points: Points fixes [(x,y,z), ...]
        forces: Forces appliquées [(x,y,z, Fx,Fy,Fz), ...]
        
    Returns:
        Dict contenant les résultats de l'analyse ou les erreurs
    """
    # Valeurs par défaut
    if fixed_points is None:
        fixed_points = [(0, 0, 0)]  # Point fixe à l'origine par défaut
    if forces is None:
        forces = [(0, 0, 0, 0, -9.81, 0)]  # Force de gravité par défaut
        
    try:
        # Obtenir les propriétés du matériau
        materials = MaterialProperties.get_default_materials()
        if material_name not in materials:
            return {"error": f"Matériau '{material_name}' non supporté"}
        material = materials[material_name]
        
        # Créer et configurer l'analyse FEM
        analysis = FEMAnalysis(mesh_path, material)
        
        if not analysis.load_mesh():
            return {"error": "Échec du chargement du maillage"}
            
        if not analysis.setup_problem(fixed_points, forces):
            return {"error": "Échec de la configuration du problème"}
            
        if not analysis.solve():
            return {"error": "Échec de la résolution"}
            
        # Obtenir et retourner les résultats
        results = analysis.get_results()
        
        # Ajouter des recommandations basées sur les résultats
        if "max_stress" in results:
            if results["min_safety_factor"] < 1.5:
                results["recommendations"] = [
                    "Le facteur de sécurité est trop faible",
                    "Considérez renforcer la structure",
                    f"Essayez un matériau plus résistant que {material_name}"
                ]
            elif results["max_displacement"] > 1.0:  # déplacement > 1mm
                results["recommendations"] = [
                    "Les déformations sont importantes",
                    "Envisagez d'ajouter des supports"
                ]
            else:
                results["recommendations"] = ["Le modèle semble structurellement sain"]
                
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test avec un modèle simple
    test_model = "path/to/test/model.stl"
    if Path(test_model).exists():
        results = analyze_model(test_model)
        print("Résultats de l'analyse FEM:")
        print(results)