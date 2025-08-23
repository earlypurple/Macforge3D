import sys
import os

# Ajouter le dossier parent au chemin Python pour pouvoir importer nos modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.text_to_mesh import create_text_mesh

def test_text_to_3d():
    # Créer un dossier pour nos tests s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Examples/text_3d_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Tester avec différents textes
    texts = [
        "Hello 3D",
        "MacForge",
        "Test"
    ]
    
    for text in texts:
        output_file = os.path.join(output_dir, f"{text.replace(' ', '_')}.obj")
        print(f"\nCréation du modèle 3D pour le texte : '{text}'")
        
        try:
            result = create_text_mesh(
                text=text,
                font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Using DejaVu Sans font
                output_path=output_file,
                font_size=48,
                depth=10.0  # Profondeur du texte en 3D
            )
            
            if isinstance(result, str) and result.startswith("Error:"):
                print(f"❌ Erreur : {result}")
            elif result and os.path.exists(output_file):
                print(f"✅ Modèle créé avec succès : {output_file}")
            else:
                print(f"❌ Erreur lors de la création du modèle pour '{text}' - Résultat: {result}")
                
        except Exception as e:
            import traceback
            print(f"❌ Erreur : {str(e)}")
            print("Détails de l'erreur:")
            print(traceback.format_exc())

if __name__ == "__main__":
    print("Test de génération de texte en 3D...")
    test_text_to_3d()
