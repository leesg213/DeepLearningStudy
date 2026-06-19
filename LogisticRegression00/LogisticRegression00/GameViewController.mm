//
//  GameViewController.m
//  LogisticRegression00
//
//  Created by Sugu Lee on 6/19/26.
//

#import "GameViewController.h"
#import "CatClassifier.hpp"
#include <vector>

#define TRAINING_IMAGE_WIDTH 64

@implementation GameViewController
{
    MTKView *_view;
    CatClassifier *_classifier;
    IBOutlet NSImageView* imageView;
    IBOutlet NSTextField* statusLabel;
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
}

- (IBAction)startTraining:(id)sender
{
    NSLog(@"Start Training button clicked!");
    
    // Disable button during training
    self.trainButton.enabled = NO;
    [self.trainButton setTitle:@"Training..."];
    
    // Collect all image paths first
    NSMutableArray *imagePaths = [NSMutableArray array];
    
    [self walkFilesAtPath:@"/Users/sugulee/Documents/Datasets/CatImages" withFileHandler:^(NSString *filePath, BOOL isDirectory) {
        
        if(isDirectory)
        {
            return;
        }
        
        NSString *extension = [[filePath pathExtension] lowercaseString];
        bool isImage = [extension isEqualToString:@"jpg"] ||
        [extension isEqualToString:@"jpeg"] ||
        [extension isEqualToString:@"png"];
        
        if(!isImage)
        {
            return;
        }
        
        [imagePaths addObject:filePath];
    }];
    
    NSLog(@"Found %lu images to process", (unsigned long)imagePaths.count);
    
    // Process images one by one with a delay to visualize
    __block NSInteger currentIndex = 0;
    __block dispatch_block_t processNextImage;
    __block std::vector<uint8_t> trainingData;
    __block std::vector<std::vector<uint8_t>> allTrainingData;
    __block std::vector<uint8_t> isCatData;
    trainingData.resize(TRAINING_IMAGE_WIDTH*TRAINING_IMAGE_WIDTH*3);
    
    processNextImage = ^{
        if (currentIndex < imagePaths.count && currentIndex<1000) {
            NSString *imagePath = imagePaths[currentIndex];
            
            // Load and display the image
            [self loadImageFromPath:imagePath];
            [statusLabel setStringValue:[NSString stringWithFormat:@"%d/%d", currentIndex, imagePaths.count]];
            
            [self extractRGBDataFromImage:imageView.image :trainingData];
            
            allTrainingData.push_back(trainingData);
            isCatData.push_back(1);
            
            //_classifier->Train(trainingData, true);
            
            currentIndex++;
            
            // Schedule next image (adjust delay as needed - 0.1 seconds here)
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.000 * NSEC_PER_SEC)),
                          dispatch_get_main_queue(), processNextImage);
        } else {
            
            _classifier->Train(allTrainingData, isCatData, 1000, 0.009);
            
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

- (void)dealloc
{
    delete _classifier;
}

@end
