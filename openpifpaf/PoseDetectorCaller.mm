//
//  Caller.m
//  openpifpafios
//
//  Created by Alexander Ivkin on 3/25/24.
//

#import "PoseDetectorCaller.h"
#import <Libtorch-Lite/Libtorch-Lite.h>
#import <CoreGraphics/CoreGraphics.h>
#import "InferredPose.h"

using namespace std;
using namespace at;

@implementation PoseDetectorCaller {
@protected torch::jit::mobile::Module _model;
}

- (nullable instancetype)initWithModelAt:(NSString*)modelPath {
    self = [super init];
    if (self) {
        try {
            if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
                NSLog(@"Model file not found at path: %@", modelPath);
                return nil;
            }
            NSLog(@"Loading model from path: %@", modelPath);
            self->_model = torch::jit::_load_for_mobile(modelPath.UTF8String);
            NSLog(@"Loaded model with name: %s", _model.name().c_str());
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

Tensor normalize(Tensor tensor) {
    const float mean[] = {0.485, 0.456, 0.406};
    const float stdV[] = {0.229, 0.224, 0.225};
    for (int i = 0; i < 3; i++) {
        tensor[0][i] = (tensor[0][i] / 255 - mean[i]) / stdV[i];
    }
    return tensor;
}

Tensor pad16(Tensor tensor) {
    return torch::constant_pad_nd(tensor, {0, 15, 0, 15}, 0);
}

Tensor tensorFromImage(CGImageRef image) {
    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);
    size_t channels = 3;
    size_t bytesPerRow = CGImageGetBytesPerRow(image);
    size_t bitsPerComponent = CGImageGetBitsPerComponent(image);
    size_t bytesPerImage = bytesPerRow * height;
    size_t totalBytes = bytesPerImage * channels;
    auto data = new uint8_t[totalBytes];
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(data, width, height, bitsPerComponent, bytesPerRow, colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    auto tensor = torch::from_blob(data, {1, static_cast<long long>(height), static_cast<long long>(width), static_cast<long long>(channels)}, at::kByte);
    delete[] data;
    tensor = tensor.permute({0, 3, 1, 2});
    NSLog(@"Loaded tensor from image");
    tensor = normalize(tensor);
    NSLog(@"Normalized tensor");
    tensor = pad16(tensor);
    NSLog(@"Padded tensor");
    return tensor.toType(at::kFloat);
}

- (InferredPose*) infer:(CGImageRef)image {
    Tensor tensor = tensorFromImage(image);
    return [self doInfer:tensor];
}

NSArray<NSNumber*>* getArray(Tensor tensor) {
    float* floatArray = tensor.data_ptr<float>();
    return [NSArray arrayWithObjects:@(floatArray[0]), @(floatArray[1]), nil];
}

- (InferredPose*) doInfer:(Tensor)tensor {
    try {
        vector<Tensor> imageList = { tensor };
        IValue input = IValue(tensor);
        NSLog(@"Running pytorch model...");
        c10::InferenceMode guard;
        auto tuple = _model.forward({input}).toTuple();
        NSLog(@"Inference done. Reading results...");
        auto results = tuple->elements()[0].toTuple();
        auto annotation = results->elements()[0].toTensor();
        cout << annotation << endl;
        auto xs = getArray(annotation.index({0}));
        auto ys = getArray(annotation.index({1}));
        auto scores = getArray(annotation.index({2}));
        return [[InferredPose alloc] initWithXs:xs ys:ys scores:scores];
    } catch (const std::exception& exception) {
        NSLog(@"Error during inference %s", exception.what());
    }
    return {};
}

@end
