// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MacForge3D",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "MacForge3D",
            targets: ["MacForge3D"])
    ],
    dependencies: [
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", .upToNextMajor(from: "0.9.0"))
    ],
    targets: [
        .executableTarget(
            name: "MacForge3D",
            dependencies: ["ZIPFoundation"],
            path: "MacForge3D"
        ),
        .testTarget(
            name: "MacForge3DTests",
            dependencies: ["MacForge3D"],
            path: "Tests/MacForge3DTests"
        )
    ]
)
