// Writes an Apple spatial photo (stereo-pair HEIC) from a left and a right image.
//
// usage: spatial_writer LEFT RIGHT OUT.heic FOCAL_PX BASELINE_M [DISPARITY_ADJUSTMENT]
//
// The pair is tagged as a stereo group with per-eye camera intrinsics and
// extrinsics, which is what Photos on Apple Vision Pro / iPhone looks for.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
guard args.count >= 6 else {
    FileHandle.standardError.write("usage: spatial_writer LEFT RIGHT OUT.heic FOCAL_PX BASELINE_M [DISPARITY_ADJUSTMENT]\n".data(using: .utf8)!)
    exit(2)
}

func loadImage(_ path: String) -> CGImage {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        FileHandle.standardError.write("cannot read \(path)\n".data(using: .utf8)!)
        exit(1)
    }
    return image
}

let left = loadImage(args[1])
let right = loadImage(args[2])
let output = URL(fileURLWithPath: args[3])
let focal = Double(args[4])!
let baseline = Double(args[5])!
let disparityAdjustment = args.count > 6 ? Double(args[6])! : 0.0

let width = Double(left.width)
let height = Double(left.height)
let intrinsics: [Double] = [focal, 0, width / 2, 0, focal, height / 2, 0, 0, 1]

func properties(isLeft: Bool) -> [CFString: Any] {
    let eyeKey = isLeft ? kCGImagePropertyGroupImageIsLeftImage : kCGImagePropertyGroupImageIsRightImage
    return [
        kCGImagePropertyGroups: [
            kCGImagePropertyGroupIndex: 0,
            kCGImagePropertyGroupType: kCGImagePropertyGroupTypeStereoPair,
            eyeKey: true,
            kCGImagePropertyGroupImageDisparityAdjustment: Int((disparityAdjustment * 1e4).rounded()),
        ],
        kCGImagePropertyHEIFDictionary: [
            kIIOMetadata_CameraModelKey: [
                kIIOCameraModel_Intrinsics: intrinsics,
                kIIOCameraModel_ModelType: kIIOCameraModelType_SimplifiedPinhole,
            ],
            kIIOMetadata_CameraExtrinsicsKey: [
                kIIOCameraExtrinsics_CoordinateSystemID: 0,
                kIIOCameraExtrinsics_Position: [isLeft ? 0.0 : baseline, 0.0, 0.0],
                kIIOCameraExtrinsics_Rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ],
        kCGImageDestinationLossyCompressionQuality: 0.92,
    ]
}

guard let destination = CGImageDestinationCreateWithURL(output as CFURL, UTType.heic.identifier as CFString, 2, nil) else {
    FileHandle.standardError.write("cannot create \(output.path)\n".data(using: .utf8)!)
    exit(1)
}
CGImageDestinationAddImage(destination, left, properties(isLeft: true) as CFDictionary)
CGImageDestinationAddImage(destination, right, properties(isLeft: false) as CFDictionary)
guard CGImageDestinationFinalize(destination) else {
    FileHandle.standardError.write("failed to write \(output.path)\n".data(using: .utf8)!)
    exit(1)
}
