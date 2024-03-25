//
//  PoseView.swift
//  openpifpafios
//
//  Created by Sven Kreiss on 12.11.20.
//

import CoreGraphics
import SwiftUI

struct Skeleton: Shape {
    let pose: Pose
    let cThreshold: Double
    
    func path(in rect: CGRect) -> Path {
        var path = Path()

        for connection in pose.skeleton {
            let from = pose.keypoints[connection[0]]
            let to = pose.keypoints[connection[1]]
            if from.c < cThreshold || to.c < cThreshold {
                continue
            }
            path.move(to: CGPoint(x: from.x, y: from.y))
            path.addLine(to: CGPoint(x: to.x, y: to.y))
        }
        
        return path
    }
}

struct PoseView: View {
    let kpSize: CGFloat = 10.0
    let cThreshold: Double = 0.5
    @Binding var poses: [Pose]
    
    var body: some View {
        let xScale = UIScreen.main.bounds.width
        let yScale = UIScreen.main.bounds.height
        
        ZStack {
            ForEach(poses, id: \.id) { pose in
                let pixelPoints = pose.keypoints.map { Keypoint(c: $0.c, x: $0.x * xScale, y: $0.y * yScale) }
                let pose = Pose(keypoints: pixelPoints, skeleton: pose.skeleton)
                Skeleton(pose: pose, cThreshold: cThreshold)
                    .stroke(lineWidth: 5)
                    .foregroundColor(.green)
                ForEach(0 ..< pose.keypoints.count) { keypoint_i in
                    let keypoint = pose.keypoints[keypoint_i]
                    if keypoint.c > cThreshold {
                        Circle()
                            .fill(Color.orange)
                            .frame(width: kpSize, height: kpSize)
                            .position(x: CGFloat(keypoint.x), y: CGFloat(keypoint.y))
                    }
                }
            }
        }
    }
}

struct PoseView_Previews: PreviewProvider {
    static var previews: some View {
        let examplePose = Pose(
            keypoints: [
                Keypoint(c: 1.0, x: 0.5, y: 0.1),
                Keypoint(c: 1.0, x: 0.8, y: 0.1),
                Keypoint(c: 1.0, x: 0.3, y: 0.9)
            ],
            skeleton: [[0, 1], [1, 2]]
        )
        PoseView(poses: .constant([examplePose]))
    }
}
