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
#import "cifcaf.hpp"

using namespace std;
using namespace at;
using namespace openpifpaf::decoder;

@implementation PoseDetectorCaller {
@protected torch::jit::mobile::Module _model;
@protected CifCaf * _cifcaf;
    
@private int cif_stride;
@private int caf_stride;
@private Tensor _skeleton;
}

Tensor getSkeleton() {
    vector<pair<int64_t, int64_t>> data = {
        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
        {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
        {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7},   {16, 20}, {16, 19},
        {16, 18}, {17, 23}, {17, 21}, {17, 22}, {25, 5},  {39, 4},  {54, 1},
        {60, 3},  {3, 63},  {66, 2},  {2, 69},  {24, 25}, {25, 26}, {26, 27},
        {27, 28}, {28, 29}, {29, 30}, {30, 31}, {31, 32}, {32, 33}, {33, 34},
        {34, 35}, {35, 36}, {36, 37}, {37, 38}, {38, 39}, {39, 40}, {24, 41},
        {41, 42}, {42, 43}, {43, 44}, {44, 45}, {45, 51}, {40, 50}, {50, 49},
        {49, 48}, {48, 47}, {47, 46}, {46, 51}, {24, 60}, {60, 61}, {61, 62},
        {62, 63}, {63, 51}, {63, 64}, {64, 65}, {65, 60}, {40, 69}, {69, 68},
        {68, 67}, {67, 66}, {66, 51}, {66, 71}, {71, 70}, {70, 69}, {51, 52},
        {52, 53}, {53, 54}, {54, 55}, {55, 56}, {56, 57}, {57, 58}, {58, 59},
        {59, 54}, {57, 75}, {78, 36}, {72, 28}, {72, 83}, {72, 73}, {73, 74},
        {74, 75}, {75, 76}, {76, 77}, {77, 78}, {78, 79}, {79, 80}, {80, 81},
        {81, 82}, {82, 83}, {72, 84}, {84, 85}, {85, 86}, {86, 87}, {87, 88},
        {88, 78}, {72, 91}, {91, 90}, {90, 89}, {89, 78}, {92, 10}, {92, 93},
        {92, 97}, {92, 101}, {92, 105}, {92, 109}, {93, 94}, {94, 95}, {95, 96},
        {97, 98}, {98, 99}, {99, 100}, {101, 102}, {102, 103}, {103, 104},
        {105, 106}, {106, 107}, {107, 108}, {109, 110}, {110, 111}, {111, 112},
        {94, 97}, {97, 101}, {101, 105}, {105, 109}, {113, 11}, {113, 114},
        {113, 118}, {113, 122}, {113, 126}, {113, 130}, {114, 115}, {115, 116},
        {116, 117}, {118, 119}, {119, 120}, {120, 121}, {122, 123}, {123, 124},
        {124, 125}, {126, 127}, {127, 128}, {128, 129}, {130, 131}, {131, 132},
        {132, 133}, {115, 118}, {118, 122}, {122, 126}, {126, 130}
    };
    auto t = torch::from_blob(data.data(), {static_cast<int64_t>(data.size()), 2}, torch::kInt64);
    t = t - 1;
    return t;
}


- (nullable instancetype)initWithModelAt:(NSString*)modelPath {
    self = [super init];
    if (self) {
        cif_stride = 8;
        caf_stride = 8;
        _skeleton = getSkeleton();
        try {
            if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
                NSLog(@"Model file not found at path: %@", modelPath);
                return nil;
            }
            NSLog(@"Loading model from path: %@", modelPath);
            self->_model = torch::jit::_load_for_mobile(modelPath.UTF8String);
            NSLog(@"Loaded model with name: %s", _model.name().c_str());
            
            _cifcaf = new CifCaf(133, _skeleton);
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
        tensor[i] = (tensor[i] - mean[i]) / stdV[i];
    }
    return tensor;
}

Tensor centerPad(Tensor tensor, double multiple) {
    int h = tensor.size(1);
    int w = tensor.size(2);

    int target_width = std::ceil((w - 1) / multiple) * multiple + 1;
    int target_height = std::ceil((h - 1) / multiple) * multiple + 1;
    
    int left = std::max(0, static_cast<int>((target_width - w) / 2.0));
    int top = std::max(0, static_cast<int>((target_height - h) / 2.0));

    int right = std::max(0, target_width - w - left);
    int bottom = std::max(0, target_height - h - top);
    cout << "padding with " << left << " " << top << " " << right << " " << bottom << endl;

    auto channel0 = torch::constant_pad_nd(tensor[0], {left, top, right, bottom}, 124);
    auto channel1 = torch::constant_pad_nd(tensor[1], {left, top, right, bottom}, 116);
    auto channel2 = torch::constant_pad_nd(tensor[2], {left, top, right, bottom}, 104);
    auto padded = torch::stack({channel0, channel1, channel2});
    return padded;
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
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image);
    CGContextRef context = CGBitmapContextCreate(data, width, height, bitsPerComponent, bytesPerRow, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    auto tensor = torch::from_blob(data, {static_cast<long long>(height), static_cast<long long>(width), static_cast<long long>(channels)}, at::kByte);
    delete[] data;
    NSLog(@"Loaded tensor from image");
    for (int i = 100; i < 110; i++) {
        cout << "[";
        for (int j = 10; j < 13; j++) {
            cout << "[";
            for (int c = 0; c < 3; c++) {
                cout << tensor[i][j][c].item<int>() << " ";
            }
            cout << "]";
        }
        cout << "]" << endl;
    }
    tensor = tensor.permute({2, 0, 1});
    tensor = centerPad(tensor, 16);
    cout << "Padded tensor size " << tensor.sizes() << endl;
    for (int i = 100; i < 120; i++) {
        cout << "[";
        for (int j = 10; j < 13; j++) {
            cout << "[";
            for (int c = 0; c < 3; c++) {
                cout << tensor[c][i][j].item<int>() << " ";
            }
            cout << "]";
        }
        cout << "]" << endl;
    }
    tensor = tensor.toType(c10::kFloat).mul(1.0 / 255);
    tensor = normalize(tensor);
    cout << "Normalized tensor" << endl;
    for (int i = 100; i < 110; i++) {
        cout << "[";
        for (int j = 10; j < 13; j++) {
            cout << "[";
            for (int c = 0; c < 3; c++) {
                cout << tensor[c][i][j].item<float>() << " ";
            }
            cout << "]";
        }
        cout << "]" << endl;
    }
    return tensor;
}

- (InferredPose*) infer:(CGImageRef)image {
    Tensor tensor = tensorFromImage(image).unsqueeze(0);
    return [self doInfer:tensor];
}

NSArray<NSNumber*>* getArray(Tensor tensor) {
    float* floatArray = tensor.data_ptr<float>();
    return [NSArray arrayWithObjects:@(floatArray[0]), @(floatArray[1]), nil];
}

- (nullable InferredPose*) doInfer:(Tensor)tensor {
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        vector<Tensor> imageList = { tensor };
        IValue input = IValue(tensor);
        NSLog(@"Running pytorch model...");
        c10::InferenceMode guard;
        auto tuple = _model.forward({input}).toTuple();
        auto inference_end_time = std::chrono::high_resolution_clock::now();
        NSLog(@"Inference done in %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(inference_end_time - start_time).count());
        auto cif_field = tuple->elements()[0].toTensor().squeeze(0);
        cout << "CIF field shape: " << cif_field.sizes() << endl;
        cout << "cif field: " << cif_field[0][0][1] << endl;
        auto caf_field = tuple->elements()[1].toTensor().squeeze(0);
        cout << "CAF field shape: " << caf_field.sizes() << endl;
        auto result = _cifcaf->call(cif_field, cif_stride, caf_field, caf_stride);
        NSLog(@"CIFCAF done in %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - inference_end_time).count());
        auto annotation = get<0>(result);
        cout << "Annotation shape: " << annotation.sizes() << endl;
        if (annotation.size(0) == 0) {
            NSLog(@"No pose detected");
            return nil;
        }
        auto xs = getArray(annotation.index({0}));
        auto ys = getArray(annotation.index({1}));
        auto scores = getArray(annotation.index({2}));
        return [[InferredPose alloc] initWithXs:xs ys:ys scores:scores];
    } catch (const std::exception& exception) {
        NSLog(@"Error during inference %s", exception.what());
    }
    return nil;
}

@end
