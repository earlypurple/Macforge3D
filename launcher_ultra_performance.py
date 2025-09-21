#!/usr/bin/env python3
"""
MacForge3D Ultra-Performance Launcher
Interface native macOS avec performances SolidWorks-level
"""

import sys
import os
import time
import threading
import numpy as np
from pathlib import Path

# Ajouter le chemin Python au sys.path
current_dir = Path(__file__).parent
python_dir = current_dir / "Python"
sys.path.insert(0, str(python_dir))

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    from ai_models.ultra_performance_engine import UltraPerformanceEngine, test_ultra_performance
    from ai_models.realtime_renderer import RealTimeRenderer, RealTimeViewer, RenderSettings
    from ai_models.simple_generator import Simple3DGenerator
    ENGINE_AVAILABLE = True
except ImportError as e:
    ENGINE_AVAILABLE = False
    engine_error = str(e)

class MacForge3DLauncher:
    """
    Lanceur MacForge3D Ultra-Performance
    
    Fonctionnalités :
    - Interface native macOS
    - Performances temps réel
    - Génération IA avancée
    - Rendu professionnel
    - Monitoring performance
    """
    
    def __init__(self):
        if not TKINTER_AVAILABLE:
            print("❌ tkinter non disponible")
            sys.exit(1)
        
        # Configuration de la fenêtre principale
        self.root = tk.Tk()
        self.root.title("🍎 MacForge3D Ultra-Performance - v2.0")
        self.root.geometry("1200x800")
        
        # Style macOS natif
        self.setup_macos_style()
        
        # Initialisation des moteurs
        self.performance_engine = None
        self.renderer = None
        self.generator = None
        self.is_engines_loaded = False
        
        # Métriques temps réel
        self.performance_metrics = {
            'fps': 0.0,
            'vertices_per_sec': 0.0,
            'memory_mb': 0.0,
            'gpu_usage': 0.0
        }
        
        # Interface utilisateur
        self.setup_ui()
        
        # Chargement asynchrone des moteurs
        self.load_engines_async()
        
        print("🚀 MacForge3D Ultra-Performance Launcher initialisé")
    
    def setup_macos_style(self):
        """Configuration style natif macOS."""
        # Couleurs macOS
        self.colors = {
            'bg': '#f0f0f0',
            'primary': '#007AFF',
            'success': '#34C759',
            'warning': '#FF9500',
            'danger': '#FF3B30',
            'text': '#1D1D1F',
            'secondary': '#8E8E93'
        }
        
        # Configuration de la fenêtre
        self.root.configure(bg=self.colors['bg'])
        
        # Style TTK
        style = ttk.Style()
        style.theme_use('default')
        
        # Configuration des styles
        style.configure('Title.TLabel', 
                       font=('SF Pro Display', 24, 'bold'),
                       foreground=self.colors['text'],
                       background=self.colors['bg'])
        
        style.configure('Subtitle.TLabel',
                       font=('SF Pro Display', 14),
                       foreground=self.colors['secondary'],
                       background=self.colors['bg'])
        
        style.configure('Action.TButton',
                       font=('SF Pro Display', 12, 'bold'),
                       foreground='white')
        
        style.map('Action.TButton',
                 background=[('active', self.colors['primary']),
                           ('pressed', '#0056CC'),
                           ('!pressed', self.colors['primary'])])
    
    def setup_ui(self):
        """Configuration de l'interface utilisateur."""
        # Frame principal avec padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration grille
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # En-tête
        self.setup_header(main_frame)
        
        # Section statut
        self.setup_status_section(main_frame)
        
        # Section actions principales
        self.setup_actions_section(main_frame)
        
        # Section performance
        self.setup_performance_section(main_frame)
        
        # Section sortie/logs
        self.setup_output_section(main_frame)
        
        # Mise à jour périodique
        self.update_performance_display()
    
    def setup_header(self, parent):
        """Section en-tête."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Titre principal
        title_label = ttk.Label(header_frame, text="🍎 MacForge3D Ultra-Performance",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Sous-titre
        subtitle_label = ttk.Label(header_frame, 
                                  text="Générateur 3D Professionnel - Performances SolidWorks-level",
                                  style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, sticky=tk.W)
        
        # Version et statut
        version_frame = ttk.Frame(header_frame)
        version_frame.grid(row=0, column=1, sticky=tk.E)
        
        version_label = ttk.Label(version_frame, text="v2.0", 
                                 font=('SF Pro Display', 12, 'bold'),
                                 foreground=self.colors['primary'])
        version_label.pack(side=tk.RIGHT)
    
    def setup_status_section(self, parent):
        """Section statut système."""
        status_frame = ttk.LabelFrame(parent, text="🔍 Statut Système", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Indicateurs de statut
        self.status_labels = {}
        
        statuses = [
            ("Moteur Performance", "engine_status"),
            ("Renderer Temps Réel", "renderer_status"), 
            ("Générateur IA", "generator_status"),
            ("Cache Optimisé", "cache_status")
        ]
        
        for i, (name, key) in enumerate(statuses):
            # Label nom
            name_label = ttk.Label(status_frame, text=f"{name}:")
            name_label.grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=(0, 10))
            
            # Label statut
            status_label = ttk.Label(status_frame, text="⏳ Chargement...",
                                   foreground=self.colors['warning'])
            status_label.grid(row=i//2, column=(i%2)*2+1, sticky=tk.W, padx=(0, 20))
            
            self.status_labels[key] = status_label
    
    def setup_actions_section(self, parent):
        """Section actions principales."""
        actions_frame = ttk.LabelFrame(parent, text="🚀 Actions Ultra-Performance", padding="15")
        actions_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Configuration grille
        actions_frame.columnconfigure(0, weight=1)
        actions_frame.columnconfigure(1, weight=1)
        actions_frame.columnconfigure(2, weight=1)
        
        # Boutons d'action
        self.action_buttons = {}
        
        actions = [
            ("🎨 Générateur IA", self.launch_ai_generator, 0, 0),
            ("⚡ Test Performance", self.run_performance_test, 0, 1),
            ("🎮 Viewer 3D", self.launch_3d_viewer, 0, 2),
            ("🔧 Optimiseur Mesh", self.launch_mesh_optimizer, 1, 0),
            ("🌊 Surface Paramétrique", self.generate_parametric_surface, 1, 1),
            ("📊 Benchmark Complet", self.run_full_benchmark, 1, 2)
        ]
        
        for text, command, row, col in actions:
            btn = ttk.Button(actions_frame, text=text, command=command,
                           style='Action.TButton', width=20)
            btn.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E))
            self.action_buttons[text] = btn
            btn.configure(state='disabled')  # Désactivé jusqu'au chargement
    
    def setup_performance_section(self, parent):
        """Section monitoring performance."""
        perf_frame = ttk.LabelFrame(parent, text="📊 Performance Temps Réel", padding="10")
        perf_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Configuration grille
        perf_frame.columnconfigure(0, weight=1)
        perf_frame.columnconfigure(1, weight=1)
        perf_frame.columnconfigure(2, weight=1)
        perf_frame.columnconfigure(3, weight=1)
        
        # Métriques de performance
        self.perf_labels = {}
        
        metrics = [
            ("FPS", "fps", "fps"),
            ("Vertices/sec", "vertices_per_sec", "vertices"),
            ("Mémoire", "memory_mb", "MB"),
            ("GPU", "gpu_usage", "%")
        ]
        
        for i, (name, key, unit) in enumerate(metrics):
            # Label nom métrique
            name_label = ttk.Label(perf_frame, text=f"{name}:",
                                  font=('SF Pro Display', 10, 'bold'))
            name_label.grid(row=0, column=i, sticky=tk.W, padx=5)
            
            # Label valeur
            value_label = ttk.Label(perf_frame, text=f"0 {unit}",
                                   font=('SF Pro Display', 12, 'bold'),
                                   foreground=self.colors['primary'])
            value_label.grid(row=1, column=i, sticky=tk.W, padx=5)
            
            self.perf_labels[key] = value_label
    
    def setup_output_section(self, parent):
        """Section sortie et logs."""
        output_frame = ttk.LabelFrame(parent, text="📝 Console de Sortie", padding="10")
        output_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Configuration grille
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(4, weight=1)
        
        # Zone de texte avec scrollbar
        text_frame = ttk.Frame(output_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.output_text = tk.Text(text_frame, height=15, width=80,
                                  font=('Monaco', 10),
                                  bg='#1E1E1E', fg='#FFFFFF',
                                  insertbackground='white')
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        # Boutons de contrôle
        control_frame = ttk.Frame(output_frame)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        clear_btn = ttk.Button(control_frame, text="🗑️ Clear", command=self.clear_output)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        save_btn = ttk.Button(control_frame, text="💾 Sauvegarder", command=self.save_output)
        save_btn.pack(side=tk.LEFT)
    
    def load_engines_async(self):
        """Chargement asynchrone des moteurs."""
        def load_worker():
            try:
                self.log("🔄 Chargement des moteurs ultra-performance...")
                
                if ENGINE_AVAILABLE:
                    # Moteur de performance
                    self.log("⚡ Initialisation moteur performance...")
                    self.performance_engine = UltraPerformanceEngine()
                    self.update_status("engine_status", "✅ Actif", self.colors['success'])
                    
                    # Générateur simple
                    self.log("🎨 Initialisation générateur IA...")
                    self.generator = Simple3DGenerator()
                    self.update_status("generator_status", "✅ Actif", self.colors['success'])
                    
                    # Renderer temps réel
                    self.log("🎮 Initialisation renderer temps réel...")
                    settings = RenderSettings(width=800, height=600, target_fps=60)
                    self.renderer = RealTimeRenderer(settings)
                    self.update_status("renderer_status", "✅ Actif", self.colors['success'])
                    
                    # Cache optimisé
                    self.log("💾 Cache optimisé activé...")
                    self.update_status("cache_status", "✅ Actif", self.colors['success'])
                    
                    self.is_engines_loaded = True
                    
                    # Activation des boutons
                    self.root.after(0, self.enable_action_buttons)
                    
                    self.log("🎉 Tous les moteurs chargés avec succès!")
                    self.log("🚀 MacForge3D Ultra-Performance prêt!")
                    
                else:
                    self.log(f"❌ Erreur chargement moteurs: {engine_error}")
                    self.update_status("engine_status", "❌ Erreur", self.colors['danger'])
                    self.update_status("renderer_status", "❌ Erreur", self.colors['danger'])
                    self.update_status("generator_status", "❌ Erreur", self.colors['danger'])
                    self.update_status("cache_status", "❌ Erreur", self.colors['danger'])
                
            except Exception as e:
                self.log(f"❌ Erreur critique: {e}")
                import traceback
                self.log(traceback.format_exc())
        
        # Lancement du thread de chargement
        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()
    
    def enable_action_buttons(self):
        """Active les boutons d'action."""
        for button in self.action_buttons.values():
            button.configure(state='normal')
    
    def update_status(self, key, text, color):
        """Met à jour un indicateur de statut."""
        if key in self.status_labels:
            self.status_labels[key].configure(text=text, foreground=color)
    
    def log(self, message):
        """Ajoute un message au log."""
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        # Thread-safe update
        self.root.after(0, lambda: self._append_to_output(full_message))
        print(message)  # Console aussi
    
    def _append_to_output(self, message):
        """Ajoute du texte à la sortie (thread-safe)."""
        self.output_text.insert(tk.END, message)
        self.output_text.see(tk.END)
    
    def clear_output(self):
        """Efface la sortie."""
        self.output_text.delete(1.0, tk.END)
    
    def save_output(self):
        """Sauvegarde la sortie dans un fichier."""
        content = self.output_text.get(1.0, tk.END)
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(content)
            self.log(f"💾 Sortie sauvegardée: {filename}")
    
    def update_performance_display(self):
        """Met à jour l'affichage des performances."""
        if self.is_engines_loaded and self.performance_engine:
            try:
                # Récupération des métriques
                report = self.performance_engine.get_performance_report()
                metrics = report.get('current_metrics', {})
                
                # Mise à jour de l'affichage
                fps = getattr(metrics, 'render_fps', 0.0)
                vertices_per_sec = getattr(metrics, 'vertices_per_second', 0.0)
                memory_mb = getattr(metrics, 'memory_usage_mb', 0.0)
                gpu_usage = getattr(metrics, 'gpu_utilization', 0.0)
                
                self.perf_labels['fps'].configure(text=f"{fps:.1f} fps")
                self.perf_labels['vertices_per_sec'].configure(text=f"{vertices_per_sec:,.0f} v/s")
                self.perf_labels['memory_mb'].configure(text=f"{memory_mb:.1f} MB")
                self.perf_labels['gpu_usage'].configure(text=f"{gpu_usage:.1f} %")
                
            except Exception as e:
                pass  # Ignore errors silently
        
        # Programmation de la prochaine mise à jour
        self.root.after(1000, self.update_performance_display)
    
    # Actions des boutons
    
    def launch_ai_generator(self):
        """Lance le générateur IA."""
        if not self.is_engines_loaded:
            messagebox.showwarning("Attention", "Moteurs en cours de chargement...")
            return
        
        def generate_worker():
            try:
                self.log("🎨 Lancement générateur IA...")
                
                # Génération d'un modèle test
                prompt = "un cube élégant avec des détails"
                model = self.generator.generate_from_text(prompt)
                
                # Sauvegarde
                filename = f"generated_model_{int(time.time())}.obj"
                if self.generator.save_obj(model, filename):
                    self.log(f"✅ Modèle généré et sauvé: {filename}")
                else:
                    self.log("❌ Erreur sauvegarde")
                
            except Exception as e:
                self.log(f"❌ Erreur générateur: {e}")
        
        thread = threading.Thread(target=generate_worker, daemon=True)
        thread.start()
    
    def run_performance_test(self):
        """Lance un test de performance."""
        if not self.is_engines_loaded:
            messagebox.showwarning("Attention", "Moteurs en cours de chargement...")
            return
        
        def test_worker():
            try:
                self.log("🏁 Lancement test de performance...")
                
                # Test d'optimisation
                vertices = np.random.rand(50000, 3) * 10
                faces = np.random.randint(0, 50000, (100000, 3))
                
                start_time = time.time()
                opt_vertices, opt_faces = self.performance_engine.optimize_mesh(vertices, faces, 'high')
                optimization_time = time.time() - start_time
                
                vertices_per_sec = len(vertices) / optimization_time if optimization_time > 0 else 0
                
                self.log(f"✅ Optimisation: {len(vertices)} → {len(opt_vertices)} vertices")
                self.log(f"⚡ Performance: {vertices_per_sec:,.0f} vertices/seconde")
                self.log(f"⏱️  Temps: {optimization_time:.3f} secondes")
                
            except Exception as e:
                self.log(f"❌ Erreur test performance: {e}")
        
        thread = threading.Thread(target=test_worker, daemon=True)
        thread.start()
    
    def launch_3d_viewer(self):
        """Lance le viewer 3D."""
        if not self.is_engines_loaded:
            messagebox.showwarning("Attention", "Moteurs en cours de chargement...")
            return
        
        def viewer_worker():
            try:
                self.log("🎮 Lancement viewer 3D temps réel...")
                
                # Création du viewer
                viewer = RealTimeViewer(width=800, height=600)
                
                # Ajout de modèles de démonstration
                # Cube
                cube_vertices = np.array([
                    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
                ])
                cube_faces = np.array([
                    [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
                    [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
                    [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
                ])
                
                viewer.add_model(cube_vertices, cube_faces, np.array([255, 100, 100]))
                
                self.log("✅ Viewer 3D lancé avec succès")
                viewer.run()
                
            except Exception as e:
                self.log(f"❌ Erreur viewer 3D: {e}")
        
        thread = threading.Thread(target=viewer_worker, daemon=True)
        thread.start()
    
    def launch_mesh_optimizer(self):
        """Lance l'optimiseur de mesh."""
        if not self.is_engines_loaded:
            messagebox.showwarning("Attention", "Moteurs en cours de chargement...")
            return
        
        def optimizer_worker():
            try:
                self.log("🔧 Lancement optimiseur mesh ultra...")
                
                # Génération d'un mesh complexe
                complex_mesh = self.performance_engine.generate_advanced_geometry(
                    'cad_primitive',
                    primitive_type='sphere',
                    radius=2.0,
                    u_segments=128,
                    v_segments=64
                )
                
                vertices = complex_mesh['vertices']
                faces = complex_mesh['faces']
                
                self.log(f"📊 Mesh original: {len(vertices)} vertices, {len(faces)} faces")
                
                # Optimisation ultra
                opt_vertices, opt_faces = self.performance_engine.optimize_mesh(
                    vertices, faces, 'ultra'
                )
                
                reduction_vertices = 100 * (1 - len(opt_vertices) / len(vertices))
                reduction_faces = 100 * (1 - len(opt_faces) / len(faces))
                
                self.log(f"✅ Mesh optimisé: {len(opt_vertices)} vertices, {len(opt_faces)} faces")
                self.log(f"📉 Réduction vertices: {reduction_vertices:.1f}%")
                self.log(f"📉 Réduction faces: {reduction_faces:.1f}%")
                
            except Exception as e:
                self.log(f"❌ Erreur optimiseur: {e}")
        
        thread = threading.Thread(target=optimizer_worker, daemon=True)
        thread.start()
    
    def generate_parametric_surface(self):
        """Génère une surface paramétrique."""
        if not self.is_engines_loaded:
            messagebox.showwarning("Attention", "Moteurs en cours de chargement...")
            return
        
        def surface_worker():
            try:
                self.log("🌊 Génération surface paramétrique avancée...")
                
                # Surface de tore complexe
                torus_surface = self.performance_engine.generate_advanced_geometry(
                    'parametric_surface',
                    surface_function='torus',
                    u_steps=120,
                    v_steps=60
                )
                
                vertices = torus_surface['vertices']
                faces = torus_surface['faces']
                
                self.log(f"✅ Surface générée: {len(vertices)} vertices, {len(faces)} faces")
                self.log(f"🎯 Type: {torus_surface['function']}")
                self.log(f"📏 Résolution: {torus_surface['resolution']}")
                
                # Sauvegarde
                filename = f"parametric_surface_{int(time.time())}.obj"
                self.save_mesh_obj(vertices, faces, filename)
                
            except Exception as e:
                self.log(f"❌ Erreur surface paramétrique: {e}")
        
        thread = threading.Thread(target=surface_worker, daemon=True)
        thread.start()
    
    def run_full_benchmark(self):
        """Lance un benchmark complet."""
        if not self.is_engines_loaded:
            messagebox.showwarning("Attention", "Moteurs en cours de chargement...")
            return
        
        def benchmark_worker():
            try:
                self.log("🏁 Benchmark complet MacForge3D Ultra-Performance...")
                self.log("=" * 60)
                
                # Benchmark du moteur de performance
                results = self.performance_engine.benchmark_performance('large')
                
                self.log("📊 RÉSULTATS BENCHMARK:")
                self.log(f"⚡ Optimisation: {results['optimization_time']:.3f}s")
                self.log(f"🎨 Génération: {results['generation_time']:.3f}s")
                self.log(f"🌊 Surface: {results['surface_generation_time']:.3f}s")
                self.log(f"📈 Rate optimisation: {results['optimization_rate']:,.0f} vertices/sec")
                
                # Test de rendu
                self.log("\n🎮 Test rendu temps réel...")
                
                # Modèles de test
                models = []
                
                # Sphère haute résolution
                sphere = self.performance_engine.generate_advanced_geometry(
                    'cad_primitive',
                    primitive_type='sphere',
                    radius=2.0,
                    u_segments=64,
                    v_segments=32
                )
                models.append(sphere)
                
                # Cylindre complexe
                cylinder = self.performance_engine.generate_advanced_geometry(
                    'cad_primitive',
                    primitive_type='cylinder',
                    radius=1.5,
                    height=3.0,
                    segments=48
                )
                models.append(cylinder)
                
                # Test de rendu
                lights = [{'type': 'directional', 'direction': np.array([1, 1, -1]),
                          'color': np.array([255, 255, 255]), 'intensity': 0.8}]
                
                render_times = []
                for i in range(5):
                    start_time = time.time()
                    frame = self.renderer.render_scene(models, lights)
                    render_time = time.time() - start_time
                    render_times.append(render_time)
                    
                    if i == 0:
                        self.log(f"🖼️  Frame {i+1}: {render_time*1000:.1f}ms")
                
                avg_render_time = np.mean(render_times)
                avg_fps = 1.0 / avg_render_time if avg_render_time > 0 else 0
                
                self.log(f"🎯 FPS moyen: {avg_fps:.1f}")
                self.log(f"⏱️  Temps rendu moyen: {avg_render_time*1000:.1f}ms")
                
                # Évaluation générale
                if avg_fps >= 60 and results['optimization_rate'] >= 100000:
                    grade = "EXCELLENT - Niveau SolidWorks ✅"
                elif avg_fps >= 30 and results['optimization_rate'] >= 50000:
                    grade = "TRÈS BON - Niveau Professionnel 🚀"
                elif avg_fps >= 15:
                    grade = "BON - Performances acceptables 👍"
                else:
                    grade = "À AMÉLIORER - Optimisations nécessaires ⚠️"
                
                self.log(f"\n🏆 ÉVALUATION GLOBALE: {grade}")
                self.log("=" * 60)
                self.log("🎉 Benchmark terminé!")
                
            except Exception as e:
                self.log(f"❌ Erreur benchmark: {e}")
        
        thread = threading.Thread(target=benchmark_worker, daemon=True)
        thread.start()
    
    def save_mesh_obj(self, vertices, faces, filename):
        """Sauvegarde un mesh au format OBJ."""
        try:
            with open(filename, 'w') as f:
                f.write(f"# MacForge3D Ultra-Performance\n")
                f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Vertices
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                # Faces
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            self.log(f"💾 Mesh sauvegardé: {filename}")
            return True
        except Exception as e:
            self.log(f"❌ Erreur sauvegarde: {e}")
            return False
    
    def run(self):
        """Lance l'application."""
        self.log("🍎 MacForge3D Ultra-Performance Launcher")
        self.log("Performances niveau SolidWorks pour macOS")
        self.log("=" * 50)
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log("👋 Fermeture de MacForge3D...")
        except Exception as e:
            self.log(f"❌ Erreur critique: {e}")

def main():
    """Point d'entrée principal."""
    print("🚀 Lancement MacForge3D Ultra-Performance...")
    
    try:
        launcher = MacForge3DLauncher()
        launcher.run()
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
