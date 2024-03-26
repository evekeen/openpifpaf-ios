//
//  InferredPose.h
//  openpifpafios
//
//  Created by Alexander Ivkin on 3/25/24.
//

#ifndef InferredPose_h
#define InferredPose_h

#import <Foundation/Foundation.h>

@interface InferredPose : NSObject
    @property (nonatomic, strong) NSArray<NSNumber *> *xs;
    @property (nonatomic, strong) NSArray<NSNumber *> *ys;
    @property (nonatomic, strong) NSArray<NSNumber *> *scores;

    - (instancetype)initWithXs:(NSArray<NSNumber *> *)xs ys:(NSArray<NSNumber *> *)ys scores:(NSArray<NSNumber *> *)scores;
@end


#endif /* InferredPose_h */
