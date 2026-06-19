//
//  GameViewController.m
//  LogisticRegression00
//
//  Created by Sugu Lee on 6/19/26.
//

#import "GameViewController.h"
#import "CatClassifier.hpp"

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
    
    processNextImage = ^{
        if (currentIndex < imagePaths.count) {
            NSString *imagePath = imagePaths[currentIndex];
            
            // Load and display the image
            [self loadImageFromPath:imagePath];
            [statusLabel setStringValue:[NSString stringWithFormat:@"%d/%d", currentIndex, imagePaths.count]];
            
            currentIndex++;
            
            // Schedule next image (adjust delay as needed - 0.1 seconds here)
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.000 * NSEC_PER_SEC)),
                          dispatch_get_main_queue(), processNextImage);
        } else {
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

- (void)dealloc
{
    delete _classifier;
}

@end
