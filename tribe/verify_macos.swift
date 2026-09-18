// Decode through AVFoundation independently of the FFmpeg model-input path.
import Foundation
import AVFoundation
import AppKit

do {
    let asset = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    let image = try generator.copyCGImage(at: .zero, actualTime: nil)
    let bitmap = NSBitmapImageRep(cgImage: image)
    try bitmap.representation(using: .png, properties: [:])!.write(
        to: URL(fileURLWithPath: CommandLine.arguments[2]))
} catch {
    fputs("Native video decode failed: \(error)\n", stderr)
    exit(1)
}
