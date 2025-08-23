import SwiftUI
import PythonKit

struct AnimationEditor: View {
    @State private var selectedAnimationType = "rotation"
    @State private var duration: Double = 2.0
    @State private var loop: Bool = true
    @State private var selectedAxis = "Y"
    @State private var amplitude: Double = 1.0
    @State private var frequency: Double = 1.0
    @State private var height: Double = 5.0
    @State private var damping: Double = 0.3
    @State private var numBounces: Int = 2
    @State private var isPlaying: Bool = false
    @State private var currentTime: Double = 0.0
    @State private var selectedKeyframe: Int? = nil
    @State private var keyframes: [KeyframeData] = []
    @State private var showExportSheet = false
    @State private var exportFormat = "glTF"
    
    private let animationTypes = ["rotation", "wave", "bounce", "translation", "scale"]
    private let axes = ["X", "Y", "Z"]
    private let exportFormats = ["glTF", "FBX"]
    
    struct KeyframeData: Identifiable {
        let id = UUID()
        var time: Double
        var value: [Double]
        var easing: String
    }
    
    var body: some View {
        HSplitView {
            // Panneau de contrôle
            VStack(alignment: .leading, spacing: 20) {
                // Type d'animation
                GroupBox("Type d'Animation") {
                    Picker("Type", selection: $selectedAnimationType) {
                        ForEach(animationTypes, id: \.self) { type in
                            Text(type.capitalized)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }
                
                // Paramètres communs
                GroupBox("Paramètres Généraux") {
                    VStack(alignment: .leading) {
                        HStack {
                            Text("Durée:")
                            Slider(value: $duration, in: 0.1...10.0)
                            Text("\(duration, specifier: "%.1f")s")
                        }
                        Toggle("Boucle", isOn: $loop)
                    }
                }
                
                // Paramètres spécifiques au type
                GroupBox("Paramètres Spécifiques") {
                    VStack(alignment: .leading) {
                        switch selectedAnimationType {
                        case "rotation":
                            Picker("Axe", selection: $selectedAxis) {
                                ForEach(axes, id: \.self) { axis in
                                    Text(axis)
                                }
                            }
                            .pickerStyle(SegmentedPickerStyle())
                            
                        case "wave":
                            HStack {
                                Text("Amplitude:")
                                Slider(value: $amplitude, in: 0.1...5.0)
                                Text("\(amplitude, specifier: "%.1f")")
                            }
                            HStack {
                                Text("Fréquence:")
                                Slider(value: $frequency, in: 0.1...5.0)
                                Text("\(frequency, specifier: "%.1f")Hz")
                            }
                            Picker("Axe", selection: $selectedAxis) {
                                ForEach(axes, id: \.self) { axis in
                                    Text(axis)
                                }
                            }
                            .pickerStyle(SegmentedPickerStyle())
                            
                        case "bounce":
                            HStack {
                                Text("Hauteur:")
                                Slider(value: $height, in: 0.1...10.0)
                                Text("\(height, specifier: "%.1f")")
                            }
                            HStack {
                                Text("Amortissement:")
                                Slider(value: $damping, in: 0.0...1.0)
                                Text("\(damping, specifier: "%.2f")")
                            }
                            Stepper("Nombre de rebonds: \(numBounces)", value: $numBounces, in: 1...10)
                            
                        default:
                            EmptyView()
                        }
                    }
                }
                
                // Timeline
                GroupBox("Timeline") {
                    VStack {
                        // Contrôles de lecture
                        HStack {
                            Button(action: { isPlaying.toggle() }) {
                                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                            }
                            Button(action: { currentTime = 0 }) {
                                Image(systemName: "backward.end.fill")
                            }
                            Slider(value: $currentTime, in: 0...duration)
                            Text("\(currentTime, specifier: "%.2f")s")
                        }
                        
                        // Liste des keyframes
                        List(keyframes) { keyframe in
                            HStack {
                                Text("T: \(keyframe.time, specifier: "%.2f")s")
                                Spacer()
                                Text("[\(keyframe.value.map { String(format: "%.1f", $0) }.joined(separator: ", "))]")
                                Text(keyframe.easing)
                            }
                            .background(selectedKeyframe == keyframes.firstIndex(where: { $0.id == keyframe.id }) ? Color.blue.opacity(0.2) : Color.clear)
                            .onTapGesture {
                                selectedKeyframe = keyframes.firstIndex(where: { $0.id == keyframe.id })
                            }
                        }
                        .frame(height: 150)
                    }
                }
                
                // Actions
                HStack {
                    Button("Ajouter Keyframe") {
                        addKeyframe()
                    }
                    
                    Button("Supprimer Keyframe") {
                        removeSelectedKeyframe()
                    }
                    .disabled(selectedKeyframe == nil)
                    
                    Spacer()
                    
                    Button("Exporter") {
                        showExportSheet = true
                    }
                }
            }
            .padding()
            .frame(minWidth: 300)
            
            // Prévisualisation 3D
            GeometryReader { geometry in
                ZStack {
                    // Votre vue 3D ici
                    Color.black
                        .overlay(
                            Text("Prévisualisation 3D")
                                .foregroundColor(.white)
                        )
                }
            }
        }
        .sheet(isPresented: $showExportSheet) {
            VStack(spacing: 20) {
                Text("Exporter l'Animation")
                    .font(.title)
                
                Picker("Format", selection: $exportFormat) {
                    ForEach(exportFormats, id: \.self) { format in
                        Text(format)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                
                Toggle("Optimiser les keyframes", isOn: .constant(true))
                Toggle("Inclure les textures", isOn: .constant(true))
                
                HStack {
                    Button("Annuler") {
                        showExportSheet = false
                    }
                    
                    Button("Exporter") {
                        exportAnimation()
                        showExportSheet = false
                    }
                }
                .padding()
            }
            .padding()
            .frame(width: 400)
        }
    }
    
    // MARK: - Méthodes
    
    private func addKeyframe() {
        var value: [Double]
        switch selectedAnimationType {
        case "rotation":
            value = [0, 0, 0]
            value[axes.firstIndex(of: selectedAxis)!] = Double.pi
        case "wave", "bounce":
            value = [0, 0, 0]
            value[axes.firstIndex(of: selectedAxis)!] = amplitude
        default:
            value = [0, 0, 0]
        }
        
        let newKeyframe = KeyframeData(
            time: currentTime,
            value: value,
            easing: "linear"
        )
        
        keyframes.append(newKeyframe)
        keyframes.sort { $0.time < $1.time }
    }
    
    private func removeSelectedKeyframe() {
        if let selected = selectedKeyframe {
            keyframes.remove(at: selected)
            selectedKeyframe = nil
        }
    }
    
    private func exportAnimation() {
        let exporter = PythonObject.import("ai_models.animation_exporter")
        
        let settings = exporter.ExportSettings(
            format: exportFormat.lowercased(),
            optimize_keyframes: true,
            embed_textures: true
        )
        
        let animator = PythonObject.import("ai_models.text_animator").TextAnimator()
        
        // Convertir les keyframes en format Python
        let pyKeyframes = keyframes.map { kf in
            exporter.AnimationKeyframe(
                time: kf.time,
                value: kf.value,
                easing: kf.easing
            )
        }
        
        // Créer l'animation
        let animationClip = exporter.AnimationClip(
            type: selectedAnimationType,
            keyframes: pyKeyframes,
            duration: duration,
            loop: loop
        )
        
        animator.add_animation("main", animationClip)
        
        // Exporter
        let exportPath = "/tmp/animation_export.\(exportFormat.lowercased())"
        exporter.AnimationExporter(settings).export_animation(
            animator,
            exportPath
        )
    }
}

#Preview {
    AnimationEditor()
}
