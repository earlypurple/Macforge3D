import SwiftUI
import UniformTypeIdentifiers

struct ExportButton: View {
    let results: [String: Any]
    @State private var isExporting = false
    @State private var selectedFormat = "json"
    @State private var error: String?
    
    private let formats = [
        ("JSON", "json"),
        ("CSV", "csv"),
        ("VTK", "vtk"),
        ("HTML", "html")
    ]
    
    var body: some View {
        Menu {
            ForEach(formats, id: \.0) { format in
                Button(format.0) {
                    exportResults(format: format.1)
                }
            }
        } label: {
            Label(
                "Exporter",
                systemImage: "square.and.arrow.up"
            )
        }
        .alert("Erreur d'export", isPresented: .constant(error != nil)) {
            Button("OK") {
                error = nil
            }
        } message: {
            if let error = error {
                Text(error)
            }
        }
    }
    
    private func exportResults(format: String) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json, .commaSeparatedText, .xml, .html]
        panel.nameFieldStringValue = "simulation_results"
        
        panel.begin { response in
            if response == .OK {
                guard let url = panel.url else { return }
                
                DispatchQueue.global(qos: .background).async {
                    do {
                        let exporter = SimulationExporter(
                            output_dir: url.deletingLastPathComponent().path
                        )
                        _ = try exporter.export_results(
                            results: results,
                            format: format,
                            filename: url.lastPathComponent
                        )
                    } catch {
                        DispatchQueue.main.async {
                            self.error = error.localizedDescription
                        }
                    }
                }
            }
        }
    }
}

struct ComparisonView: View {
    let results1: [String: Any]
    let results2: [String: Any]
    @StateObject private var viewModel = ComparisonViewModel()
    
    var body: some View {
        VStack {
            if viewModel.isLoading {
                ProgressView("Comparaison en cours...")
            } else if let comparison = viewModel.comparison {
                ScrollView {
                    VStack(spacing: 20) {
                        // Différences principales
                        MainDifferencesSection(comparison: comparison)
                        
                        // Tests statistiques
                        StatisticalTestsSection(comparison: comparison)
                        
                        // Graphiques de comparaison
                        ComparisonChartsSection(
                            comparison: comparison,
                            results1: results1,
                            results2: results2
                        )
                        
                        // Corrélations
                        CorrelationsSection(comparison: comparison)
                    }
                    .padding()
                }
            } else if let error = viewModel.error {
                Text(error)
                    .foregroundColor(.red)
                    .padding()
            }
        }
        .onAppear {
            viewModel.compareResults(results1, results2)
        }
    }
}

class ComparisonViewModel: ObservableObject {
    @Published var comparison: [String: Any]?
    @Published var isLoading = false
    @Published var error: String?
    
    private let comparator = SimulationComparator()
    
    func compareResults(_ results1: [String: Any], _ results2: [String: Any]) {
        isLoading = true
        error = nil
        
        DispatchQueue.global(qos: .background).async {
            do {
                let comparison = self.comparator.compare_results(
                    results1,
                    results2
                )
                DispatchQueue.main.async {
                    self.comparison = comparison
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.error = error.localizedDescription
                    self.isLoading = false
                }
            }
        }
    }
}

struct MainDifferencesSection: View {
    let comparison: [String: Any]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Différences principales")
                .font(.title2)
                .bold()
            
            ForEach(metrics, id: \.self) { metric in
                if let diff = differences[metric] as? [String: Any] {
                    DifferenceRow(
                        metric: metric,
                        difference: diff
                    )
                }
            }
        }
    }
    
    private var metrics: [String] {
        (comparison["metrics"] as? [String: Any])?.keys.sorted() ?? []
    }
    
    private var differences: [String: Any] {
        comparison["metrics"] as? [String: Any] ?? [:]
    }
}

struct DifferenceRow: View {
    let metric: String
    let difference: [String: Any]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(metric)
                .font(.headline)
            
            HStack {
                VStack(alignment: .leading) {
                    StatValue(
                        label: "Différence moyenne",
                        value: difference["difference_mean"] as? Double ?? 0
                    )
                    StatValue(
                        label: "RMSE",
                        value: difference["rmse"] as? Double ?? 0
                    )
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    StatValue(
                        label: "Corrélation",
                        value: difference["correlation"] as? Double ?? 0
                    )
                    StatValue(
                        label: "Différence relative",
                        value: difference["relative_difference"] as? Double ?? 0,
                        format: "%.1f%%"
                    )
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

struct StatValue: View {
    let label: String
    let value: Double
    var format: String = "%.3f"
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(String(format: format, value))
                .font(.system(.body, design: .monospaced))
        }
    }
}

struct StatisticalTestsSection: View {
    let comparison: [String: Any]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Tests statistiques")
                .font(.title2)
                .bold()
            
            ForEach(tests.keys.sorted(), id: \.self) { metric in
                if let test = tests[metric] as? [String: Any] {
                    TestResultRow(
                        metric: metric,
                        test: test
                    )
                }
            }
        }
    }
    
    private var tests: [String: Any] {
        comparison["statistical_tests"] as? [String: Any] ?? [:]
    }
}

struct TestResultRow: View {
    let metric: String
    let test: [String: Any]
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(metric)
                    .font(.headline)
                Text(test["test"] as? String ?? "")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing) {
                Text(String(format: "p = %.3f", test["p_value"] as? Double ?? 0))
                    .font(.system(.body, design: .monospaced))
                
                Text(
                    test["significant"] as? Bool == true
                        ? "Significatif"
                        : "Non significatif"
                )
                .font(.caption)
                .foregroundColor(
                    test["significant"] as? Bool == true
                        ? .green
                        : .secondary
                )
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

struct ComparisonChartsSection: View {
    let comparison: [String: Any]
    let results1: [String: Any]
    let results2: [String: Any]
    @State private var selectedMetric: String?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Graphiques comparatifs")
                .font(.title2)
                .bold()
            
            if let metrics = (comparison["metrics"] as? [String: Any])?.keys {
                Picker("Métrique", selection: $selectedMetric) {
                    ForEach(Array(metrics), id: \.self) { metric in
                        Text(metric).tag(metric as String?)
                    }
                }
                .pickerStyle(.menu)
                
                if let metric = selectedMetric {
                    ComparisonChart(
                        results1: results1,
                        results2: results2,
                        metric: metric
                    )
                    .frame(height: 300)
                }
            }
        }
    }
}

struct ComparisonChart: View {
    let results1: [String: Any]
    let results2: [String: Any]
    let metric: String
    
    var body: some View {
        Chart {
            ForEach(values1.indices, id: \.self) { i in
                LineMark(
                    x: .value("Index", i),
                    y: .value("Valeur 1", values1[i])
                )
                .foregroundStyle(.blue)
                
                LineMark(
                    x: .value("Index", i),
                    y: .value("Valeur 2", values2[i])
                )
                .foregroundStyle(.red)
            }
        }
    }
    
    private var values1: [Double] {
        getValues(from: results1)
    }
    
    private var values2: [Double] {
        getValues(from: results2)
    }
    
    private func getValues(from results: [String: Any]) -> [Double] {
        let keys = metric.split(separator: ".")
        var current = results
        
        for key in keys.dropLast() {
            current = current[String(key)] as? [String: Any] ?? [:]
        }
        
        if let values = current[String(keys.last!)] as? [Double] {
            return values
        }
        return []
    }
}

struct CorrelationsSection: View {
    let comparison: [String: Any]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Corrélations")
                .font(.title2)
                .bold()
            
            ForEach(correlations.keys.sorted(), id: \.self) { metric in
                if let corr = correlations[metric] as? [String: Any] {
                    CorrelationRow(
                        metric: metric,
                        correlation: corr
                    )
                }
            }
        }
    }
    
    private var correlations: [String: Any] {
        comparison["correlations"] as? [String: Any] ?? [:]
    }
}

struct CorrelationRow: View {
    let metric: String
    let correlation: [String: Any]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(metric)
                .font(.headline)
            
            HStack {
                VStack(alignment: .leading) {
                    if let pearson = correlation["pearson"] as? [String: Any] {
                        Text("Pearson: \(formatCorrelation(pearson))")
                    }
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    if let spearman = correlation["spearman"] as? [String: Any] {
                        Text("Spearman: \(formatCorrelation(spearman))")
                    }
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
    
    private func formatCorrelation(_ corr: [String: Any]) -> String {
        if let r = corr["correlation"] as? Double,
           let p = corr["p_value"] as? Double {
            return String(format: "r = %.3f (p = %.3f)", r, p)
        }
        return "N/A"
    }
}
