import AppKit
import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit

private let protocolVersion = "screen-capture-kit-stream-v1"
private let frameMagic = Data("SCF1".utf8)

private final class CaptureOutput: NSObject, SCStreamOutput, SCStreamDelegate,
    @unchecked Sendable
{
    private let lock = NSLock()
    private var latestPixelBuffer: CVPixelBuffer?
    private(set) var streamError: String?
    let callbackQueue = DispatchQueue(
        label: "doudizhu.screen-capture-kit.frames",
        qos: .userInteractive
    )

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .screen, sampleBuffer.isValid else {
            return
        }
        let attachments = CMSampleBufferGetSampleAttachmentsArray(
            sampleBuffer,
            createIfNecessary: false
        ) as? [[SCStreamFrameInfo: Any]]
        let statusRaw = attachments?.first?[SCStreamFrameInfo.status] as? Int
        let status = statusRaw.flatMap(SCFrameStatus.init(rawValue:))
        if status == .blank || status == .suspended || status == .stopped {
            lock.lock()
            latestPixelBuffer = nil
            streamError = "target window is minimized, blank, or unavailable"
            lock.unlock()
            return
        }
        guard status == .complete || status == .started,
              sampleBuffer.dataReadiness == .ready,
              let buffer = sampleBuffer.imageBuffer
        else {
            return
        }
        lock.lock()
        latestPixelBuffer = buffer
        streamError = nil
        lock.unlock()
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        lock.lock()
        streamError = error.localizedDescription
        lock.unlock()
    }

    func snapshot(timeoutSeconds: Double = 3.0) -> CVPixelBuffer? {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        repeat {
            lock.lock()
            let buffer = latestPixelBuffer
            let error = streamError
            lock.unlock()
            if error != nil {
                return nil
            }
            if let buffer {
                return buffer
            }
            Thread.sleep(forTimeInterval: 0.005)
        } while Date() < deadline
        return nil
    }
}

private func appendLittleEndian<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    var encoded = value.littleEndian
    withUnsafeBytes(of: &encoded) { data.append(contentsOf: $0) }
}

private func writeReadyMetadata(
    appName: String,
    window: SCWindow,
    width: Int,
    height: Int,
    fps: Int
) throws {
    let metadata: [String: Any] = [
        "protocol": protocolVersion,
        "app_name": appName,
        "window_id": Int(window.windowID),
        "window_name": window.title ?? appName,
        "window_box": [
            Int(window.frame.minX.rounded()),
            Int(window.frame.minY.rounded()),
            Int(window.frame.maxX.rounded()),
            Int(window.frame.maxY.rounded()),
        ],
        "pixel_size": [width, height],
        "fps": fps,
    ]
    var payload = try JSONSerialization.data(withJSONObject: metadata)
    payload.append(0x0A)
    FileHandle.standardOutput.write(payload)
}

private func writeFrame(_ buffer: CVPixelBuffer) throws {
    let flags = CVPixelBufferLockFlags.readOnly
    let status = CVPixelBufferLockBaseAddress(buffer, flags)
    guard status == kCVReturnSuccess,
          let baseAddress = CVPixelBufferGetBaseAddress(buffer)
    else {
        throw NSError(
            domain: "doudizhu.macos-window-stream",
            code: Int(status),
            userInfo: [NSLocalizedDescriptionKey: "cannot lock captured pixel buffer"]
        )
    }
    defer { CVPixelBufferUnlockBaseAddress(buffer, flags) }

    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let byteCount = bytesPerRow * height
    let timestampNanoseconds = UInt64(
        max(0, Date().timeIntervalSince1970 * 1_000_000_000)
    )

    var header = frameMagic
    appendLittleEndian(UInt32(width), to: &header)
    appendLittleEndian(UInt32(height), to: &header)
    appendLittleEndian(UInt32(bytesPerRow), to: &header)
    appendLittleEndian(timestampNanoseconds, to: &header)
    appendLittleEndian(UInt64(byteCount), to: &header)
    FileHandle.standardOutput.write(header)
    FileHandle.standardOutput.write(Data(bytes: baseAddress, count: byteCount))
}

private func positiveInteger(_ value: String?, default defaultValue: Int) -> Int {
    guard let value, let parsed = Int(value), parsed > 0 else {
        return defaultValue
    }
    return parsed
}

@main
private struct MacOSWindowStream {
    @MainActor
    static func main() async {
        _ = NSApplication.shared
        let arguments = CommandLine.arguments
        guard arguments.count >= 2 else {
            FileHandle.standardError.write(
                Data("usage: macos_window_stream <app-name> [fps]\n".utf8)
            )
            exit(2)
        }
        let appName = arguments[1]
        let fps = min(60, positiveInteger(arguments.count > 2 ? arguments[2] : nil, default: 12))

        do {
            let content = try await SCShareableContent.current
            let candidates = content.windows.filter {
                $0.owningApplication?.applicationName == appName
                    && $0.windowLayer == 0
                    && $0.isOnScreen
                    && $0.frame.width > 0
                    && $0.frame.height > 0
            }
            guard let window = candidates.max(by: {
                $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height
            }) else {
                throw NSError(
                    domain: "doudizhu.macos-window-stream",
                    code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "target window is not available"]
                )
            }

            let filter = SCContentFilter(desktopIndependentWindow: window)
            let configuration = SCStreamConfiguration()
            configuration.width = Int(
                filter.contentRect.width * CGFloat(filter.pointPixelScale)
            )
            configuration.height = Int(
                filter.contentRect.height * CGFloat(filter.pointPixelScale)
            )
            configuration.minimumFrameInterval = CMTime(
                value: 1,
                timescale: CMTimeScale(fps)
            )
            configuration.pixelFormat = kCVPixelFormatType_32BGRA
            configuration.queueDepth = 3
            configuration.showsCursor = false
            configuration.ignoreShadowsSingleWindow = true
            configuration.scalesToFit = true
            configuration.preservesAspectRatio = true

            let output = CaptureOutput()
            let stream = SCStream(
                filter: filter,
                configuration: configuration,
                delegate: output
            )
            try stream.addStreamOutput(
                output,
                type: .screen,
                sampleHandlerQueue: output.callbackQueue
            )
            try await stream.startCapture()
            try writeReadyMetadata(
                appName: appName,
                window: window,
                width: configuration.width,
                height: configuration.height,
                fps: fps
            )

            while let command = readLine() {
                if command == "C" {
                    guard let buffer = output.snapshot() else {
                        throw NSError(
                            domain: "doudizhu.macos-window-stream",
                            code: 4,
                            userInfo: [
                                NSLocalizedDescriptionKey:
                                    output.streamError ?? "capture stream produced no frames",
                            ]
                        )
                    }
                    try writeFrame(buffer)
                } else if command == "Q" {
                    break
                }
            }
            try await stream.stopCapture()
        } catch {
            FileHandle.standardError.write(
                Data(("macOS window stream failed: \(error.localizedDescription)\n").utf8)
            )
            exit(1)
        }
    }
}
