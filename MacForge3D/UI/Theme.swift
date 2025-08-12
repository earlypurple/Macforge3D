import SwiftUI

class ThemeManager: ObservableObject {
    @Published var darkMode: Bool = false

    func toggleDarkMode() {
        darkMode.toggle()
    }
}

// Utilisation dans l'UI :
// .preferredColorScheme(themeManager.darkMode ? .dark : .light)
