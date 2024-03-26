//
//  InferredPose.m
//  openpifpafios
//
//  Created by Alexander Ivkin on 3/25/24.
//

#import "InferredPose.h"

@implementation InferredPose

- (instancetype)initWithXs:(NSArray<NSNumber *> *)xs ys:(NSArray<NSNumber *> *)ys scores:(NSArray<NSNumber *> *)scores {
    self = [super init];
    if (self) {
        _xs = xs;
        _ys = ys;
        _scores = scores;
    }
    return self;
}

@end
