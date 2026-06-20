//
//  GameViewController.m
//  LogisticRegression00
//
//  Created by Sugu Lee on 6/19/26.
//

#import "GameViewController.h"
#import "CatClassifier.hpp"
#include <vector>
#include <H5Cpp.h>

#define TRAINING_IMAGE_WIDTH 64

@implementation GameViewController
{
    MTKView *_view;
    CatClassifier *_classifier;
    IBOutlet NSImageView* imageView, *predictImageView;
    IBOutlet NSTextField* statusLabel, *predictLabel;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    _view = (MTKView *)self.view;

    _view.device = MTLCreateSystemDefaultDevice();

    if(!_view.device)
    {
        NSLog(@"Metal is not supported on this device");
        self.view = [[NSView alloc] initWithFrame:self.view.frame];
        return;
    }
    
    // Initialize classifier
    _classifier = new CatClassifier();
    _classifier->Init(_view.device);
    
    // Set image scaling to fit the imageView frame
    imageView.imageScaling = NSImageScaleProportionallyUpOrDown;
    predictImageView.imageScaling = NSImageScaleProportionallyUpOrDown;
}
-(void)loadDataSetFromH5
{
    NSString* dataSetPath_train = @"/Users/sugulee/Documents/Files/home/jovyan/work/release/W2A2/datasets/train_catvnoncat.h5";
    NSString* dataSetPath_test = @"/Users/sugulee/Documents/Files/home/jovyan/work/release/W2A2/datasets/test_catvnoncat.h5";

    H5::H5File train_file(dataSetPath_train.UTF8String, H5F_ACC_RDONLY);
    H5::H5File test_file(dataSetPath_test.UTF8String, H5F_ACC_RDONLY);

    H5::DataSet ds_train_set_x = train_file.openDataSet("train_set_x"); // 209,64,64,3
    H5::DataSet ds_train_set_y = train_file.openDataSet("train_set_y"); // 209

    H5::DataSet ds_test_set_x = test_file.openDataSet("test_set_x");
    H5::DataSet ds_test_set_y = test_file.openDataSet("test_set_y");
    
    H5::DataSpace dataspace = ds_train_set_x.getSpace();
    
    hsize_t dims[4];
    dataspace.getSimpleExtentDims(dims, nullptr);

    int numImages = dims[0];
    int height    = dims[1];
    int width     = dims[2];
    int channels  = dims[3];
    
    std::vector<uint8_t> train_set_x(numImages * width * height * channels);
    ds_train_set_x.read(train_set_x.data(), H5::PredType::NATIVE_UINT8);
    std::vector<uint8_t> train_set_y(numImages);
    ds_train_set_y.read(train_set_y.data(), H5::PredType::NATIVE_UINT8);

    std::vector<uint8_t> test_set_x(numImages * width * height * channels);
    ds_test_set_x.read(test_set_x.data(), H5::PredType::NATIVE_UINT8);
    std::vector<uint8_t> test_set_y(numImages);
    ds_test_set_y.read(test_set_y.data(), H5::PredType::NATIVE_UINT8);

  
    NSLog(@"[DataSet loaded]");
}
-(IBAction)predict:(id)sender
{
#if false
    // Create open panel for file selection
    NSOpenPanel *openPanel = [NSOpenPanel openPanel];
    
    // Configure the panel
    openPanel.title = @"Choose an image to classify";
    openPanel.message = @"Select a JPG image file";
    openPanel.canChooseFiles = YES;
    openPanel.canChooseDirectories = NO;
    openPanel.allowsMultipleSelection = NO;
    
    // Set allowed file types
    openPanel.allowedFileTypes = @[@"jpg", @"jpeg", @"JPG", @"JPEG"];
    
    // Show the panel
    [openPanel beginSheetModalForWindow:self.view.window completionHandler:^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            // User selected a file
            NSURL *selectedURL = openPanel.URL;
            NSString *imagePath = selectedURL.path;
            
            NSLog(@"Selected image: %@", imagePath);
            
            // Load and display the selected image
            NSImage *image = [[NSImage alloc] initWithContentsOfFile:imagePath];
            if (image) {
                // Resize image to match training size
                NSImage *resizedImage = [self resizeImage:image toWidth:TRAINING_IMAGE_WIDTH height:TRAINING_IMAGE_WIDTH];
                
                // Display in predict image view
                self->predictImageView.image = resizedImage;
                
                // Extract RGB data
                std::vector<uint8_t> imageData;
                if ([self extractRGBDataFromImage:resizedImage :imageData]) {
                    // TODO: Call classifier prediction
                    float isCat = self->_classifier->Predict(imageData);
                    [self->predictLabel setStringValue:[NSString stringWithFormat:@"P : %f", isCat]];
                    
                    //[self->predictLabel setStringValue:@"Prediction ready"];
                } else {
                    [self->predictLabel setStringValue:@"Failed to process image"];
                }
            } else {
                NSLog(@"Failed to load image from: %@", imagePath);
                [self->predictLabel setStringValue:@"Failed to load image"];
            }
        } else {
            // User cancelled
            NSLog(@"Image selection cancelled");
        }
    }];
#endif
}
- (IBAction)startTraining:(id)sender
{
    NSLog(@"Start Training button clicked!");
    
    // Disable button during training
    self.trainButton.enabled = NO;
    [self.trainButton setTitle:@"Training..."];
    
    NSString* dataSetPath_train = @"/Users/sugulee/Documents/Files/home/jovyan/work/release/W2A2/datasets/train_catvnoncat.h5";
    NSString* dataSetPath_test = @"/Users/sugulee/Documents/Files/home/jovyan/work/release/W2A2/datasets/test_catvnoncat.h5";

    H5::H5File train_file(dataSetPath_train.UTF8String, H5F_ACC_RDONLY);
    H5::H5File test_file(dataSetPath_test.UTF8String, H5F_ACC_RDONLY);

    H5::DataSet ds_train_set_x = train_file.openDataSet("train_set_x"); // 209,64,64,3
    H5::DataSet ds_train_set_y = train_file.openDataSet("train_set_y"); // 209

    H5::DataSet ds_test_set_x = test_file.openDataSet("test_set_x");
    H5::DataSet ds_test_set_y = test_file.openDataSet("test_set_y");
    
    H5::DataSpace dataspace_train = ds_train_set_x.getSpace();
    H5::DataSpace dataspace_test = ds_test_set_x.getSpace();

    hsize_t dims[4];
    dataspace_train.getSimpleExtentDims(dims, nullptr);

    int numImages_train = dims[0];
    int height    = dims[1];
    int width     = dims[2];
    int channels  = dims[3];

    dataspace_test.getSimpleExtentDims(dims, nullptr);
    int numImages_test = dims[0];
    
    std::vector<uint8_t> train_set_x(numImages_train * width * height * channels);
    ds_train_set_x.read(train_set_x.data(), H5::PredType::NATIVE_UINT8);
    std::vector<uint8_t> train_set_y(numImages_train);
    ds_train_set_y.read(train_set_y.data(), H5::PredType::NATIVE_UINT8);

    std::vector<uint8_t> test_set_x(numImages_test * width * height * channels);
    ds_test_set_x.read(test_set_x.data(), H5::PredType::NATIVE_UINT8);
    std::vector<uint8_t> test_set_y(numImages_test);
    ds_test_set_y.read(test_set_y.data(), H5::PredType::NATIVE_UINT8);
    
    // Process images one by one with a delay to visualize
    __block NSInteger currentIndex = 0;
    __block dispatch_block_t processNextImage;
    __block std::vector<uint8_t> trainingData;
    __block std::vector<std::vector<uint8_t>> allTrainingData;
    __block std::vector<uint8_t> isCatData;
    trainingData.resize(TRAINING_IMAGE_WIDTH*TRAINING_IMAGE_WIDTH*3);
    
    processNextImage = ^{
        if (currentIndex < numImages_train ) {
                        
            NSImage* image = [self createImageFromRGBData:train_set_x offset:(currentIndex*(width*height*channels)) width:width height:height];
            
            imageView.image = image;
            
            [statusLabel setStringValue:[NSString stringWithFormat:@"%d/%d", currentIndex, numImages_train]];
            
            
           // allTrainingData.push_back(trainingData);
            //isCatData.push_back(isCat ? 1 : 0);
            
            //_classifier->Train(trainingData, true);
            
            currentIndex++;
            
            // Schedule next image (adjust delay as needed - 0.1 seconds here)
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.001 * NSEC_PER_SEC)),
                          dispatch_get_main_queue(), processNextImage);
        } else {
#if false
            std::vector<std::vector<uint8_t>> testTrainData;
            
            {
                std::vector<uint8_t> testImage;
                testImage.resize(2);
                testImage[0] = 1; testImage[1] = 3;
                testTrainData.push_back(testImage);
            }
            {
                std::vector<uint8_t> testImage;
                testImage.resize(2);
                testImage[0] = -2; testImage[1] = 0.5;
                testTrainData.push_back(testImage);
            }
            {
                std::vector<uint8_t> testImage;
                testImage.resize(2);
                testImage[0] = -1; testImage[1] = -3.2;
                testTrainData.push_back(testImage);
            }
            
            std::vector<uint8_t> testIsCatData;
            testIsCatData.resize(3);
            testIsCatData[0] = 1; testIsCatData[1] =1; testIsCatData[2] = 0;
            
            _classifier->Train(testTrainData, testIsCatData, 100, 0.002);
#endif
            _classifier->Train(train_set_x, train_set_y, numImages_train, width, 2000, 0.005);
            
            _classifier->Predict(train_set_x, train_set_y, numImages_train, width);
            _classifier->Predict(test_set_x, test_set_y, numImages_test, width);

            // All images processed
            NSLog(@"Training complete!");
            self.trainButton.enabled = YES;
            [self.trainButton setTitle:@"Start Training"];
        }
    };
    
    // Start processing
    processNextImage();
}

- (void)loadImageFromPath:(NSString *)imagePath
{
    if (!imagePath || imagePath.length == 0) {
        NSLog(@"Error: Image path is empty");
        return;
    }
    
    // Create image from file path
    NSImage *image = [[NSImage alloc] initWithContentsOfFile:imagePath];
    
    if (image) {
        // Display image in imageView
        imageView.image = [self resizeImage:image toWidth:TRAINING_IMAGE_WIDTH height:TRAINING_IMAGE_WIDTH];
        NSLog(@"Image loaded successfully from: %@", imagePath);
    } else {
        NSLog(@"Error: Failed to load image from path: %@", imagePath);
        imageView.image = nil;
    }
}

- (NSImage *)resizeImage:(NSImage *)sourceImage toWidth:(CGFloat)targetWidth height:(CGFloat)targetHeight
{
    if (!sourceImage) {
        NSLog(@"Error: Source image is nil");
        return nil;
    }
    
    NSSize originalSize = sourceImage.size;
    NSSize targetSize = NSMakeSize(targetWidth, targetHeight);
    
    // Calculate aspect ratios
    CGFloat originalAspect = originalSize.width / originalSize.height;
    CGFloat targetAspect = targetWidth / targetHeight;
    
    // Calculate scaled size maintaining aspect ratio
    NSSize scaledSize;
    if (originalAspect > targetAspect) {
        // Image is wider - fit to width
        scaledSize.width = targetWidth;
        scaledSize.height = targetWidth / originalAspect;
    } else {
        // Image is taller - fit to height
        scaledSize.height = targetHeight;
        scaledSize.width = targetHeight * originalAspect;
    }
    
    // Calculate position to center the image (letterbox offset)
    CGFloat xOffset = (targetWidth - scaledSize.width) / 2.0;
    CGFloat yOffset = (targetHeight - scaledSize.height) / 2.0;
    
    // Create new image with target size
    NSImage *resizedImage = [[NSImage alloc] initWithSize:targetSize];
    
    [resizedImage lockFocus];
    
    // Fill background with black color (letterbox)
    [[NSColor blackColor] setFill];
    NSRectFill(NSMakeRect(0, 0, targetWidth, targetHeight));
    
    // Draw the scaled image centered
    NSRect destRect = NSMakeRect(xOffset, yOffset, scaledSize.width, scaledSize.height);
    [sourceImage drawInRect:destRect
                   fromRect:NSZeroRect
                  operation:NSCompositingOperationSourceOver
                   fraction:1.0];
    
    [resizedImage unlockFocus];
    
    NSLog(@"Image resized from (%.0f x %.0f) to (%.0f x %.0f) with letterboxing",
          originalSize.width, originalSize.height, targetWidth, targetHeight);
    
    return resizedImage;
}

- (void)walkFilesAtPath:(NSString *)rootPath
     withFileHandler:(void (^)(NSString *filePath, BOOL isDirectory))fileHandler
{
    if (!rootPath || rootPath.length == 0) {
        NSLog(@"Error: Root path is empty");
        return;
    }
    
    if (!fileHandler) {
        NSLog(@"Error: File handler block is nil");
        return;
    }
    
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSError *error = nil;
    
    // Check if root path exists
    BOOL isDirectory = NO;
    BOOL exists = [fileManager fileExistsAtPath:rootPath isDirectory:&isDirectory];
    
    if (!exists) {
        NSLog(@"Error: Path does not exist: %@", rootPath);
        return;
    }
    
    // Use directory enumerator for recursive traversal
    NSDirectoryEnumerator *enumerator = [fileManager enumeratorAtPath:rootPath];
    
    if (!enumerator) {
        NSLog(@"Error: Failed to create enumerator for path: %@", rootPath);
        return;
    }
    
    NSString *relativePath;
    while ((relativePath = [enumerator nextObject]) != nil) {
        @autoreleasepool {
            // Build full path
            NSString *fullPath = [rootPath stringByAppendingPathComponent:relativePath];
            
            // Check if it's a directory
            BOOL itemIsDirectory = NO;
            [fileManager fileExistsAtPath:fullPath isDirectory:&itemIsDirectory];
            
            // Call the handler block
            fileHandler(fullPath, itemIsDirectory);
        }
    }
    
    NSLog(@"Finished walking files at path: %@", rootPath);
}

- (unsigned char *)extractBitmapDataFromImage:(NSImage *)image
{
    if (!image) {
        NSLog(@"Error: Image is nil");
        return NULL;
    }
    
    NSSize imageSize = image.size;
    NSInteger width = (NSInteger)imageSize.width;
    NSInteger height = (NSInteger)imageSize.height;
    
    // Create bitmap representation
    NSBitmapImageRep *bitmapRep = [[NSBitmapImageRep alloc]
                                   initWithBitmapDataPlanes:NULL
                                   pixelsWide:width
                                   pixelsHigh:height
                                   bitsPerSample:8
                                   samplesPerPixel:4  // RGBA
                                   hasAlpha:YES
                                   isPlanar:NO
                                   colorSpaceName:NSCalibratedRGBColorSpace
                                   bytesPerRow:width * 4
                                   bitsPerPixel:32];
    
    if (!bitmapRep) {
        NSLog(@"Error: Failed to create bitmap representation");
        return NULL;
    }
    
    // Draw image into bitmap context
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:[NSGraphicsContext graphicsContextWithBitmapImageRep:bitmapRep]];
    [image drawInRect:NSMakeRect(0, 0, width, height)
             fromRect:NSZeroRect
            operation:NSCompositingOperationSourceOver
             fraction:1.0];
    [NSGraphicsContext restoreGraphicsState];
    
    // Get pointer to bitmap data
    unsigned char *bitmapData = [bitmapRep bitmapData];
    
    if (!bitmapData) {
        NSLog(@"Error: Failed to get bitmap data");
        return NULL;
    }
    
    // Calculate total size (width * height * 4 bytes per pixel for RGBA)
    NSInteger dataSize = width * height * 4;
    
    // Allocate memory and copy data (caller is responsible for freeing this memory)
    unsigned char *copiedData = (unsigned char *)malloc(dataSize);
    if (copiedData) {
        memcpy(copiedData, bitmapData, dataSize);
        NSLog(@"Extracted bitmap data: %ldx%ld, %ld bytes", width, height, dataSize);
    } else {
        NSLog(@"Error: Failed to allocate memory for bitmap data");
        return NULL;
    }
    
    return copiedData;
}

- (bool)extractRGBDataFromImage:(NSImage *)image :(std::vector<uint8_t>&)outVector
{
    if (!image) {
        NSLog(@"Error: Image is nil");
        return false;
    }
    
    NSSize imageSize = image.size;
    NSInteger width = (NSInteger)imageSize.width;
    NSInteger height = (NSInteger)imageSize.height;
    
    // Create bitmap representation with RGBA (4 channels)
    NSBitmapImageRep *bitmapRep = [[NSBitmapImageRep alloc]
                                   initWithBitmapDataPlanes:NULL
                                   pixelsWide:width
                                   pixelsHigh:height
                                   bitsPerSample:8
                                   samplesPerPixel:4  // RGBA
                                   hasAlpha:YES
                                   isPlanar:NO
                                   colorSpaceName:NSCalibratedRGBColorSpace
                                   bytesPerRow:width * 4
                                   bitsPerPixel:32];
    
    if (!bitmapRep) {
        NSLog(@"Error: Failed to create bitmap representation");
        return false;
    }
    
    // Draw image into bitmap context
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:[NSGraphicsContext graphicsContextWithBitmapImageRep:bitmapRep]];
    [image drawInRect:NSMakeRect(0, 0, width, height)
             fromRect:NSZeroRect
            operation:NSCompositingOperationSourceOver
             fraction:1.0];
    [NSGraphicsContext restoreGraphicsState];
    
    // Get pointer to bitmap data (RGBA format)
    unsigned char *bitmapData = [bitmapRep bitmapData];
    
    if (!bitmapData) {
        NSLog(@"Error: Failed to get bitmap data");
        return false;
    }
    
    // Calculate RGB size (width * height * 3 bytes per pixel)
    NSInteger rgbSize = width * height * 3;
    NSInteger rgbaSize = width * height * 4;
    
    // Resize output vector
    outVector.resize(rgbSize);
    
    // Convert RGBA to RGB by stripping alpha channel
    NSInteger rgbIndex = 0;
    for (NSInteger i = 0; i < rgbaSize; i += 4) {
        outVector[rgbIndex++] = bitmapData[i];     // R
        outVector[rgbIndex++] = bitmapData[i + 1]; // G
        outVector[rgbIndex++] = bitmapData[i + 2]; // B
        // Skip alpha channel (i + 3)
    }
    
    NSLog(@"Extracted RGB data: %ldx%ld, %ld bytes", width, height, rgbSize);
    
    return true;
}

- (NSImage *)createImageFromRGBData:(const std::vector<uint8_t>&)rgbData
                            offset:(NSInteger)offset
                              width:(NSInteger)width
                             height:(NSInteger)height
{
    if (rgbData.empty()) {
        NSLog(@"Error: RGB data is empty");
        return nil;
    }
    
#if false
    NSInteger expectedSize = width * height * 3;
    if (rgbData.size() != expectedSize) {
        NSLog(@"Error: RGB data size mismatch. Expected %ld bytes, got %zu bytes",
              expectedSize, rgbData.size());
        return nil;
    }
#endif
    
    // Create bitmap representation with RGBA (we'll convert RGB to RGBA)
    NSBitmapImageRep *bitmapRep = [[NSBitmapImageRep alloc]
                                   initWithBitmapDataPlanes:NULL
                                   pixelsWide:width
                                   pixelsHigh:height
                                   bitsPerSample:8
                                   samplesPerPixel:4  // RGBA
                                   hasAlpha:YES
                                   isPlanar:NO
                                   colorSpaceName:NSCalibratedRGBColorSpace
                                   bytesPerRow:width * 4
                                   bitsPerPixel:32];
    
    if (!bitmapRep) {
        NSLog(@"Error: Failed to create bitmap representation");
        return nil;
    }
    
    // Get pointer to bitmap data
    unsigned char *bitmapData = [bitmapRep bitmapData];
    
    if (!bitmapData) {
        NSLog(@"Error: Failed to get bitmap data pointer");
        return nil;
    }
    
    // Convert RGB to RGBA by adding alpha channel
    NSInteger rgbIndex = 0;
    NSInteger rgbaIndex = 0;
    
    for (NSInteger i = 0; i < width * height; i++) {
        bitmapData[rgbaIndex++] = rgbData[offset+rgbIndex++]; // R
        bitmapData[rgbaIndex++] = rgbData[offset+rgbIndex++]; // G
        bitmapData[rgbaIndex++] = rgbData[offset+rgbIndex++]; // B
        bitmapData[rgbaIndex++] = 255;                  // A (fully opaque)
    }
    
    // Create NSImage from bitmap representation
    NSImage *image = [[NSImage alloc] initWithSize:NSMakeSize(width, height)];
    [image addRepresentation:bitmapRep];
    
    NSLog(@"Created image from RGB data: %ldx%ld", width, height);
    
    return image;
}

- (void)dealloc
{
    delete _classifier;
}

@end
