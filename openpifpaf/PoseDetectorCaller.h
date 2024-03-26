//
//  Caller.h
//  openpifpafios
//
//  Created by Alexander Ivkin on 3/25/24.
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CGImage.h>
#import "InferredPose.h"

NS_ASSUME_NONNULL_BEGIN

@interface PoseDetectorCaller : NSObject

- (nullable instancetype)initWithModelAt:(NSString*)modelPath;
- (nullable InferredPose *) infer:(CGImageRef)image;

@end

NS_ASSUME_NONNULL_END
