#!/usr/bin/env python3
"""
MacForge3D GUI Simple - Interface sans dépendances externes
Version de secours qui fonctionne avec juste tkinter
"""

import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import time
import os
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ NumPy non disponible - Mode dégradé activé")

class SimpleArray:
    """Classe simple pour remplacer numpy quand il n'est pas disponible"""
    def __init__(self, data):
        if isinstance(data, list):
            self.data = data
        else:
            self.data = list(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value

def array(data):
    """Fonction pour créer un array (numpy ou simple)"""
    if NUMPY_AVAILABLE:
        return np.array(data)
    else:
        return SimpleArray(data)

class MacForge3DSimpleGUI:
    """Interface GUI MacForge3D simplifiée sans dépendances externes"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("MacForge3D Simple GUI")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2b2b2b")
        
        # Variables
        self.scene_objects = []
        self.current_object_index = -1
        
        # Interface
        self.create_menu()
        self.create_toolbar()
        self.create_main_panels()
        self.create_statusbar()
        
        # Message de bienvenue
        self.set_status("MacForge3D Simple GUI prêt")
        
    def create_menu(self):
        """Menu principal"""
        menubar = tk.Menu(self.root, bg="#3c3c3c", fg="white")
        
        # Menu Fichier
        file_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="white")
        file_menu.add_command(label="🆕 Nouveau Projet", command=self.new_project)
        file_menu.add_command(label="📂 Ouvrir OBJ...", command=self.open_obj)
        file_menu.add_command(label="💾 Sauvegarder OBJ...", command=self.save_obj)
        file_menu.add_separator()
        file_menu.add_command(label="❌ Quitter", command=self.root.quit)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        
        # Menu Créer
        create_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="white")
        create_menu.add_command(label="📦 Cube", command=self.add_cube)
        create_menu.add_command(label="🔵 Sphère", command=self.add_sphere)
        create_menu.add_command(label="📄 Plan", command=self.add_plane)
        menubar.add_cascade(label="Créer", menu=create_menu)
        
        # Menu Outils
        tools_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="white")
        tools_menu.add_command(label="🔧 Optimiser", command=self.optimize_mesh)
        tools_menu.add_command(label="📊 Statistiques", command=self.show_stats)
        tools_menu.add_command(label="🔧 Installer Dépendances", command=self.install_dependencies)
        menubar.add_cascade(label="Outils", menu=tools_menu)
        
        self.root.config(menu=menubar)
        
    def create_toolbar(self):
        """Barre d'outils"""
        toolbar = tk.Frame(self.root, bg="#3c3c3c", height=50)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        buttons = [
            ("🆕", "Nouveau", self.new_project),
            ("📂", "Ouvrir", self.open_obj),
            ("💾", "Sauver", self.save_obj),
            ("📦", "Cube", self.add_cube),
            ("🔵", "Sphère", self.add_sphere),
            ("📄", "Plan", self.add_plane),
            ("🔧", "Optimiser", self.optimize_mesh),
            ("🗑️", "Effacer", self.clear_scene),
        ]
        
        for icon, tooltip, command in buttons:
            btn = tk.Button(toolbar, text=icon, command=command,
                          bg="#4d4d4d", fg="white", relief="flat",
                          font=("Arial", 14), width=3, height=1)
            btn.pack(side=tk.LEFT, padx=2, pady=5)
            
    def create_main_panels(self):
        """Panneaux principaux"""
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel de gauche (hiérarchie)
        left_panel = tk.Frame(main_frame, bg="#3c3c3c", width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        tk.Label(left_panel, text="🗂️ Hiérarchie des Objets", 
                bg="#3c3c3c", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.object_listbox = tk.Listbox(left_panel, bg="#4d4d4d", fg="white", 
                                       selectbackground="#00bcd4", relief="flat")
        self.object_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.object_listbox.bind('<<ListboxSelect>>', self.on_object_select)
        
        # Panel central (viewer)
        center_panel = tk.Frame(main_frame, bg="#1e1e1e")
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        viewer_header = tk.Frame(center_panel, bg="#3c3c3c", height=30)
        viewer_header.pack(fill=tk.X)
        tk.Label(viewer_header, text="🎨 Viewer 3D Simple", 
                bg="#3c3c3c", fg="white", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Zone viewer simple
        self.viewer_frame = tk.Frame(center_panel, bg="#1e1e1e")
        self.viewer_frame.pack(fill=tk.BOTH, expand=True)
        
        self.setup_simple_viewer()
        
        # Panel de droite (propriétés)
        right_panel = tk.Frame(main_frame, bg="#3c3c3c", width=250)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        tk.Label(right_panel, text="⚙️ Propriétés", 
                bg="#3c3c3c", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.properties_frame = tk.Frame(right_panel, bg="#3c3c3c")
        self.properties_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Propriétés par défaut
        self.setup_properties_panel()
        
    def setup_simple_viewer(self):
        """Viewer 3D simple en mode texte"""
        self.viewer_text = tk.Text(self.viewer_frame, bg="#1e1e1e", fg="white",
                                 font=("Monaco", 10), relief="flat")
        self.viewer_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Message initial
        welcome_text = """
🎨 MacForge3D Simple Viewer 3D

┌─────────────────────────────────────┐
│          SCÈNE 3D VIDE             │
│                                     │
│     Ajoutez des objets avec la     │
│        barre d'outils !            │
│                                     │
│   📦 Cube  🔵 Sphère  📄 Plan     │
│                                     │
└─────────────────────────────────────┘

💡 Astuce: Installez matplotlib pour le viewer 3D avancé
   Menu Outils > Installer Dépendances
"""
        self.viewer_text.insert(tk.END, welcome_text)
        self.viewer_text.config(state=tk.DISABLED)
        
        # Contrôles
        controls_frame = tk.Frame(self.viewer_frame, bg="#3c3c3c", height=40)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Button(controls_frame, text="🔄 Actualiser", command=self.update_viewer,
                 bg="#4d4d4d", fg="white", relief="flat").pack(side=tk.LEFT, padx=5, pady=5)
        
    def setup_properties_panel(self):
        """Panel des propriétés"""
        # Info objet sélectionné
        self.selected_label = tk.Label(self.properties_frame, text="Aucun objet sélectionné",
                                     bg="#3c3c3c", fg="white")
        self.selected_label.pack(pady=10)
        
        # Statistiques
        stats_frame = tk.LabelFrame(self.properties_frame, text="📊 Statistiques", 
                                  bg="#3c3c3c", fg="white")
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_label = tk.Label(stats_frame, text="Objets: 0\nVertices: 0\nFaces: 0",
                                   bg="#3c3c3c", fg="white", justify=tk.LEFT)
        self.stats_label.pack(pady=5)
        
    def create_statusbar(self):
        """Barre de statut"""
        self.status_var = tk.StringVar()
        self.status_var.set("🚀 MacForge3D Simple GUI - Prêt")
        
        statusbar = tk.Label(self.root, textvariable=self.status_var, 
                           bg="#3c3c3c", fg="white", relief="sunken", anchor="w")
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def set_status(self, message):
        """Mettre à jour le statut"""
        self.status_var.set(f"🚀 {message}")
        self.root.update_idletasks()
        
    def update_stats(self):
        """Mettre à jour les statistiques"""
        total_vertices = sum(len(obj.get('vertices', [])) for obj in self.scene_objects)
        total_faces = sum(len(obj.get('faces', [])) for obj in self.scene_objects)
        
        stats_text = f"Objets: {len(self.scene_objects)}\nVertices: {total_vertices}\nFaces: {total_faces}"
        self.stats_label.config(text=stats_text)
        
    def new_project(self):
        """Nouveau projet"""
        self.scene_objects.clear()
        self.object_listbox.delete(0, tk.END)
        self.update_viewer()
        self.set_status("Nouveau projet créé")
        
    def add_cube(self):
        """Ajouter un cube"""
        vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]
        faces = [
            [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
        ]
        
        obj = {
            'name': f'Cube_{len(self.scene_objects)+1}',
            'type': 'cube',
            'vertices': vertices,
            'faces': faces
        }
        
        self.scene_objects.append(obj)
        self.object_listbox.insert(tk.END, f"📦 {obj['name']}")
        self.update_viewer()
        self.set_status(f"Cube ajouté - {len(vertices)} vertices")
        
    def add_sphere(self):
        """Ajouter une sphère (simplifiée)"""
        # Sphère simplifiée en icosaèdre
        vertices = [
            [0, 1, 0], [0, -1, 0],  # Pôles
            [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]  # Équateur
        ]
        faces = [
            [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
            [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]
        ]
        
        obj = {
            'name': f'Sphere_{len(self.scene_objects)+1}',
            'type': 'sphere',
            'vertices': vertices,
            'faces': faces
        }
        
        self.scene_objects.append(obj)
        self.object_listbox.insert(tk.END, f"🔵 {obj['name']}")
        self.update_viewer()
        self.set_status(f"Sphère ajoutée - {len(vertices)} vertices")
        
    def add_plane(self):
        """Ajouter un plan"""
        vertices = [
            [-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]
        ]
        faces = [[0, 1, 2], [0, 2, 3]]
        
        obj = {
            'name': f'Plan_{len(self.scene_objects)+1}',
            'type': 'plane',
            'vertices': vertices,
            'faces': faces
        }
        
        self.scene_objects.append(obj)
        self.object_listbox.insert(tk.END, f"📄 {obj['name']}")
        self.update_viewer()
        self.set_status("Plan ajouté")
        
    def open_obj(self):
        """Ouvrir un fichier OBJ"""
        filename = filedialog.askopenfilename(
            title="Ouvrir un fichier OBJ",
            filetypes=[("Fichiers OBJ", "*.obj")]
        )
        
        if not filename:
            return
            
        try:
            vertices = []
            faces = []
            
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        coords = [float(x) for x in line.strip().split()[1:4]]
                        vertices.append(coords)
                    elif line.startswith('f '):
                        face_indices = []
                        for vertex in line.strip().split()[1:]:
                            idx = int(vertex.split('/')[0]) - 1
                            face_indices.append(idx)
                        if len(face_indices) >= 3:
                            faces.append(face_indices[:3])
                            
            if vertices:
                obj = {
                    'name': os.path.basename(filename),
                    'type': 'imported',
                    'vertices': vertices,
                    'faces': faces
                }
                
                self.scene_objects.append(obj)
                self.object_listbox.insert(tk.END, f"📂 {obj['name']}")
                self.update_viewer()
                self.set_status(f"OBJ importé: {len(vertices)} vertices")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'importer le fichier:\n{str(e)}")
            
    def save_obj(self):
        """Sauvegarder en OBJ"""
        if not self.scene_objects:
            messagebox.showwarning("Aucun objet", "Aucun objet à sauvegarder")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Sauvegarder en OBJ",
            defaultextension=".obj",
            filetypes=[("Fichiers OBJ", "*.obj")]
        )
        
        if not filename:
            return
            
        try:
            with open(filename, 'w') as f:
                f.write("# MacForge3D Simple GUI Export\n\n")
                
                vertex_offset = 0
                for obj in self.scene_objects:
                    vertices = obj.get('vertices', [])
                    faces = obj.get('faces', [])
                    
                    f.write(f"# {obj['name']}\n")
                    
                    for vertex in vertices:
                        f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                    
                    for face in faces:
                        face_str = " ".join(str(idx + vertex_offset + 1) for idx in face)
                        f.write(f"f {face_str}\n")
                    
                    vertex_offset += len(vertices)
                    f.write("\n")
                    
            self.set_status(f"Sauvegardé: {filename}")
            messagebox.showinfo("Export réussi", f"Fichier sauvegardé:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder:\n{str(e)}")
            
    def clear_scene(self):
        """Effacer la scène"""
        self.scene_objects.clear()
        self.object_listbox.delete(0, tk.END)
        self.update_viewer()
        self.set_status("Scène effacée")
        
    def optimize_mesh(self):
        """Optimiser le mesh"""
        if not self.scene_objects:
            messagebox.showwarning("Aucun objet", "Aucun objet à optimiser")
            return
        self.set_status("Optimisation simulée - OK")
        
    def show_stats(self):
        """Afficher les statistiques"""
        total_vertices = sum(len(obj.get('vertices', [])) for obj in self.scene_objects)
        total_faces = sum(len(obj.get('faces', [])) for obj in self.scene_objects)
        
        stats = f"""📊 Statistiques MacForge3D Simple

🔢 Objets: {len(self.scene_objects)}
⚡ Vertices: {total_vertices:,}
🔺 Faces: {total_faces:,}

🚀 Mode: Simple GUI (sans dépendances)
💡 Pour le mode avancé: Installez les dépendances"""
        
        messagebox.showinfo("Statistiques", stats)
        
    def install_dependencies(self):
        """Installer les dépendances"""
        response = messagebox.askyesno(
            "Installer Dépendances",
            "Voulez-vous installer NumPy et Matplotlib\npour le viewer 3D avancé ?\n\n"
            "Ceci peut prendre quelques minutes..."
        )
        
        if response:
            self.set_status("Installation des dépendances en cours...")
            # Ici on pourrait lancer l'installation en arrière-plan
            messagebox.showinfo("Installation", 
                              "Lancez le script Fix_Dependencies_GUI.command\n"
                              "pour installer les dépendances automatiquement.")
        
    def on_object_select(self, event):
        """Sélection d'objet"""
        selection = self.object_listbox.curselection()
        if selection:
            self.current_object_index = selection[0]
            obj = self.scene_objects[self.current_object_index]
            self.selected_label.config(text=f"Sélectionné: {obj['name']}")
            self.set_status(f"Sélectionné: {obj['name']}")
            
    def update_viewer(self):
        """Mettre à jour le viewer"""
        self.viewer_text.config(state=tk.NORMAL)
        self.viewer_text.delete(1.0, tk.END)
        
        if not self.scene_objects:
            welcome_text = """
🎨 MacForge3D Simple Viewer 3D

┌─────────────────────────────────────┐
│          SCÈNE 3D VIDE             │
│                                     │
│     Ajoutez des objets avec la     │
│        barre d'outils !            │
│                                     │
│   📦 Cube  🔵 Sphère  📄 Plan     │
│                                     │
└─────────────────────────────────────┘
"""
        else:
            viewer_content = "🎨 MacForge3D Simple Viewer 3D\n\n"
            viewer_content += "┌─────────────────────────────────────┐\n"
            viewer_content += "│            SCÈNE 3D                │\n"
            viewer_content += "├─────────────────────────────────────┤\n"
            
            for i, obj in enumerate(self.scene_objects):
                name = obj['name']
                vertices_count = len(obj.get('vertices', []))
                faces_count = len(obj.get('faces', []))
                
                if obj['type'] == 'cube':
                    icon = "📦"
                elif obj['type'] == 'sphere':
                    icon = "🔵"
                elif obj['type'] == 'plane':
                    icon = "📄"
                else:
                    icon = "📂"
                    
                viewer_content += f"│ {icon} {name:<20} │\n"
                viewer_content += f"│   Vertices: {vertices_count:<8} Faces: {faces_count:<4} │\n"
                viewer_content += "├─────────────────────────────────────┤\n"
            
            viewer_content += "└─────────────────────────────────────┘\n\n"
            viewer_content += "💡 Cliquez sur un objet dans la hiérarchie\n"
            viewer_content += "   pour le sélectionner\n\n"
            viewer_content += "🔧 Installez matplotlib pour le viewer 3D avancé"
            
            welcome_text = viewer_content
        
        self.viewer_text.insert(tk.END, welcome_text)
        self.viewer_text.config(state=tk.DISABLED)
        self.update_stats()

def main():
    """Fonction principale"""
    print("🚀 Lancement MacForge3D Simple GUI...")
    print(f"✅ Python: {sys.version}")
    
    if NUMPY_AVAILABLE:
        print("✅ NumPy disponible")
    else:
        print("⚠️ NumPy non disponible - Mode simple activé")
        print("💡 Lancez Fix_Dependencies_GUI.command pour installer les dépendances")
    
    try:
        root = tk.Tk()
        app = MacForge3DSimpleGUI(root)
        print("🎉 Interface MacForge3D Simple prête!")
        root.mainloop()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        input("Appuyez sur Entrée pour fermer...")

if __name__ == "__main__":
    main()
