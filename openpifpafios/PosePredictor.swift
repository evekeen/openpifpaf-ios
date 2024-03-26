//
//  PosePredictor.swift
//  openpifpafios
//
//  Created by Sven Kreiss on 14.11.20.
//

import CoreGraphics
import CoreML
import Foundation
import Vision

class PredictorInput: MLFeatureProvider {
    var featureNames: Set<String> {
        return ["image"]
    }
    private let image: CGImage
    private let imageSize: CGSize
    
    init(image: CGImage, size: CGSize) {
        self.image = image
        self.imageSize = size
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        return try? MLFeatureValue(cgImage: image,
                                   pixelsWide: Int(imageSize.width),
                                   pixelsHigh: Int(imageSize.height),
                                   pixelFormatType: image.pixelFormatInfo.rawValue,
                                   options: [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue])
    }
}

class PosePredictor {
//    private let model: MLModel = try! shufflenet().model
//    private let size = CGSize(width: 129, height: 97)
//    private let stride = 8.0
    var delegate: (([Pose]) -> Void)? = nil

    private lazy var model: PoseDetectorCaller = {
//        if let filePath = Bundle.main.path(forResource: "openpifpaf-shufflenetv2k16.torchscript", ofType: "ptl") {
        if let filePath = Bundle.main.path(forResource: "backbone", ofType: "ptl") {
            print("loading model from", filePath)
            return PoseDetectorCaller(modelAt: filePath)!
        } else {
            fatalError("Could not load model.")
        }
    }()
//
//    func singlePose(_ cif: MLMultiArray) -> Pose {
//        var confidences = [Double](repeating: 0.0, count: cif.shape[1].intValue)
//        var x = [Double](repeating: 0.0, count: cif.shape[1].intValue)
//        var y = [Double](repeating: 0.0, count: cif.shape[1].intValue)
//        for jIndex in 0..<cif.shape[1].intValue {
//            for yIndex in 0..<cif.shape[3].intValue {
//                for xIndex in 0..<cif.shape[4].intValue {
//                    let currentConfidence = cif[ [0, jIndex, 0, yIndex, xIndex] ]
//
//                    if currentConfidence > confidences[jIndex] {
//                        confidences[jIndex] = currentConfidence
//                        x[jIndex] = cif[ [0, jIndex, 1, yIndex, xIndex] ]
//                        y[jIndex] = cif[ [0, jIndex, 2, yIndex, xIndex] ]
//                    }
//                }
//            }
//        }
//        print("shape", cif.shape)
//        return Pose(
//            keypoints: (0..<cif.shape[1].intValue).map { Keypoint(c: confidences[$0], x: x[$0] * stride, y: y[$0] * stride) },
//            skeleton: [[0, 1], [0, 2]]
//        )
//    }
    
    func predict(_ image: CGImage) {
        print("predicting...")
        if let inferredPose = model.infer(image) {
            let pose = Pose(keypoints: inferredPose.keypoints, skeleton: [[]])
            print(pose)
            guard self.delegate != nil else { return }
            self.delegate!([pose])
        }
    }
}


// subscript for MLMultiArray expects [NSNumber] as input and returns NSNumber
// but we would rather work with [Int] and Double
extension MLMultiArray {
    subscript(index: [Int]) -> Double {
        return self[index.map { NSNumber(value: $0) } ] as! Double
    }
}

extension InferredPose {
    var keypoints: [Keypoint] {
        return (0..<xs.count).map { Keypoint(c: scores[$0].doubleValue, x: xs[$0].doubleValue, y: ys[$0].doubleValue) }
    }
}
