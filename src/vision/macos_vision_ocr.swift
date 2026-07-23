import AppKit
import Foundation
import Vision

for path in CommandLine.arguments.dropFirst() {
    var recognized = ""
    var confidence: Float = 0.0

    if let image = NSImage(contentsOfFile: path) {
        var rect = NSRect(origin: .zero, size: image.size)
        if let cgImage = image.cgImage(
            forProposedRect: &rect,
            context: nil,
            hints: nil
        ) {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.recognitionLanguages = ["en-US"]
            request.usesLanguageCorrection = false

            let handler = VNImageRequestHandler(cgImage: cgImage)
            do {
                try handler.perform([request])
                for observation in request.results ?? [] {
                    guard let candidate = observation.topCandidates(1).first else {
                        continue
                    }
                    let digits = candidate.string.filter { $0.isNumber }
                    if !digits.isEmpty && candidate.confidence >= confidence {
                        recognized = digits
                        confidence = candidate.confidence
                    }
                }
            } catch {
                recognized = ""
                confidence = 0.0
            }
        }
    }

    print("\(path)\t\(recognized)\t\(confidence)")
}
