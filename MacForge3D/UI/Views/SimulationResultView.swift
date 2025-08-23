import SwiftUI
import Charts

struct SimulationResultView: View {
    let results: [String: Any]
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Vue d'ensemble
            OverviewTab(results: results)
                .tabItem {
                    Label("Vue d'ensemble", systemImage: "chart.bar.fill")
                }
                .tag(0)
            
            // Visualisation 3D
            if let meshData = results["mesh_data"] as? [String: Any] {
                Simulation3DView(
                    meshData: meshData,
                    results: results
                )
                .tabItem {
                    Label("3D", systemImage: "cube.fill")
                }
                .tag(1)
            }
            
            // Graphiques détaillés
            ChartsTab(results: results)
                .tabItem {
                    Label("Graphiques", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(2)
            
            // Données brutes
            RawDataTab(results: results)
                .tabItem {
                    Label("Données", systemImage: "list.bullet")
                }
                .tag(3)
        }
        .frame(minWidth: 600, minHeight: 400)
    }
}

struct OverviewTab: View {
    let results: [String: Any]
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Statistiques principales
                MainStatsSection(results: results)
                
                // Graphique récapitulatif
                SummaryChartSection(results: results)
                
                // Points critiques
                CriticalPointsSection(results: results)
            }
            .padding()
        }
    }
}

struct MainStatsSection: View {
    let results: [String: Any]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Statistiques principales")
                .font(.title2)
                .bold()
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 16) {
                StatCard(
                    title: "Contrainte max",
                    value: extractValue(from: results, key: "max_stress"),
                    unit: "MPa",
                    color: .red
                )
                
                StatCard(
                    title: "Déplacement max",
                    value: extractValue(from: results, key: "max_displacement"),
                    unit: "mm",
                    color: .blue
                )
                
                StatCard(
                    title: "Température max",
                    value: extractValue(from: results, key: "max_temperature"),
                    unit: "°C",
                    color: .orange
                )
            }
        }
    }
    
    private func extractValue(from results: [String: Any], key: String) -> Double {
        (results[key] as? Double) ?? 0.0
    }
}

struct StatCard: View {
    let title: String
    let value: Double
    let unit: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(title)
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Text(String(format: "%.2f", value))
                .font(.title)
                .bold()
                .foregroundColor(color)
            
            Text(unit)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(10)
    }
}

struct SummaryChartSection: View {
    let results: [String: Any]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Résumé graphique")
                .font(.title2)
                .bold()
            
            if let data = prepareChartData() {
                Chart(data, id: \.name) { item in
                    BarMark(
                        x: .value("Valeur", item.value),
                        y: .value("Métrique", item.name)
                    )
                    .foregroundStyle(by: .value("Métrique", item.name))
                }
                .frame(height: 200)
            }
        }
    }
    
    private func prepareChartData() -> [ChartData]? {
        // Convertir les données pour le graphique
        // À adapter selon la structure des résultats
        return nil
    }
}

struct ChartData {
    let name: String
    let value: Double
}

struct CriticalPointsSection: View {
    let results: [String: Any]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Points critiques")
                .font(.title2)
                .bold()
            
            if let criticalPoints = results["critical_points"] as? [[String: Any]] {
                ForEach(criticalPoints.indices, id: \.self) { index in
                    CriticalPointRow(point: criticalPoints[index])
                }
            }
        }
    }
}

struct CriticalPointRow: View {
    let point: [String: Any]
    
    var body: some View {
        HStack {
            Circle()
                .fill(severity > 0.7 ? Color.red : Color.orange)
                .frame(width: 8, height: 8)
            
            VStack(alignment: .leading) {
                Text(description)
                    .font(.headline)
                Text("Position: \(position)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(String(format: "%.2f", value))
                .bold()
                .foregroundColor(severity > 0.7 ? .red : .orange)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
    
    private var description: String {
        point["description"] as? String ?? "Point critique"
    }
    
    private var position: String {
        if let pos = point["position"] as? [Double] {
            return String(format: "(%.1f, %.1f, %.1f)", pos[0], pos[1], pos[2])
        }
        return "N/A"
    }
    
    private var value: Double {
        point["value"] as? Double ?? 0.0
    }
    
    private var severity: Double {
        point["severity"] as? Double ?? 0.0
    }
}

struct ChartsTab: View {
    let results: [String: Any]
    @State private var selectedMetric = "stress"
    
    var body: some View {
        VStack {
            Picker("Métrique", selection: $selectedMetric) {
                Text("Contraintes").tag("stress")
                Text("Déplacements").tag("displacement")
                Text("Température").tag("temperature")
            }
            .pickerStyle(.segmented)
            .padding()
            
            TabView {
                // Distribution
                DistributionChart(data: getDistributionData())
                    .tabItem {
                        Label("Distribution", systemImage: "chart.bar.fill")
                    }
                
                // Évolution temporelle
                TimeSeriesChart(data: getTimeSeriesData())
                    .tabItem {
                        Label("Évolution", systemImage: "chart.line.uptrend.xyaxis")
                    }
                
                // Carte de chaleur
                HeatmapView(data: getHeatmapData())
                    .tabItem {
                        Label("Carte", systemImage: "square.grid.3x3.fill")
                    }
            }
        }
    }
    
    private func getDistributionData() -> [ChartData] {
        // À implémenter selon la structure des résultats
        []
    }
    
    private func getTimeSeriesData() -> [ChartData] {
        // À implémenter selon la structure des résultats
        []
    }
    
    private func getHeatmapData() -> [[Double]] {
        // À implémenter selon la structure des résultats
        []
    }
}

struct DistributionChart: View {
    let data: [ChartData]
    
    var body: some View {
        Chart(data, id: \.name) { item in
            BarMark(
                x: .value("Valeur", item.value),
                y: .value("Fréquence", item.value)
            )
        }
        .padding()
    }
}

struct TimeSeriesChart: View {
    let data: [ChartData]
    
    var body: some View {
        Chart(data, id: \.name) { item in
            LineMark(
                x: .value("Temps", item.name),
                y: .value("Valeur", item.value)
            )
        }
        .padding()
    }
}

struct HeatmapView: View {
    let data: [[Double]]
    
    var body: some View {
        // Implémentation de la carte de chaleur
        // Utiliser un Grid de rectangles colorés
        EmptyView()
    }
}

struct RawDataTab: View {
    let results: [String: Any]
    
    var body: some View {
        List {
            ForEach(Array(results.keys.sorted()), id: \.self) { key in
                Section(header: Text(key)) {
                    if let value = results[key] {
                        Text(String(describing: value))
                            .font(.system(.body, design: .monospaced))
                    }
                }
            }
        }
    }
}
