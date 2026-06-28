//
//  DataPlotView.mm
//  GoalKeeperDecision
//
//  Created by Sugu Lee on 6/28/26.
//

#import "DataPlotView.h"

@implementation DataPlotView
{
    std::vector<float> dataX;
    std::vector<uint8_t> dataLabels;
}

- (void)setDataWithX:(const std::vector<float>&)x labels:(const std::vector<uint8_t>&)labels
{
    dataX = x;
    dataLabels = labels;
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect
{
    [super drawRect:dirtyRect];
    
    // Fill background with white
    [[NSColor whiteColor] setFill];
    NSRectFill(dirtyRect);
    
    if (dataX.empty() || dataLabels.empty()) {
        return;
    }
    
    size_t numPoints = dataLabels.size();
    size_t halfSize = dataX.size() / 2;
    
    if (halfSize != numPoints) {
        NSLog(@"Data size mismatch: x array has %zu elements (expecting %zu pairs), but %zu labels", 
              dataX.size(), numPoints, numPoints);
        return;
    }
    
    // Find min/max values for scaling
    float minX = INFINITY, maxX = -INFINITY;
    float minY = INFINITY, maxY = -INFINITY;
    
    for (size_t i = 0; i < numPoints; ++i) {
        float x = dataX[i];
        float y = dataX[i + halfSize];
        
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }
    
    // Add some padding
    float rangeX = maxX - minX;
    float rangeY = maxY - minY;
    float padding = 0.1f; // 10% padding
    
    minX -= rangeX * padding;
    maxX += rangeX * padding;
    minY -= rangeY * padding;
    maxY += rangeY * padding;
    
    rangeX = maxX - minX;
    rangeY = maxY - minY;
    
    NSRect bounds = self.bounds;
    CGFloat viewWidth = NSWidth(bounds);
    CGFloat viewHeight = NSHeight(bounds);
    
    // Disc radius
    CGFloat discRadius = 4.0;
    
    // Draw each point
    for (size_t i = 0; i < numPoints; ++i) {
        float x = dataX[i];
        float y = dataX[i + halfSize];
        uint8_t label = dataLabels[i];
        
        // Normalize coordinates to [0, 1]
        float normalizedX = (x - minX) / rangeX;
        float normalizedY = (y - minY) / rangeY;
        
        // Map to view coordinates
        CGFloat viewX = normalizedX * viewWidth;
        CGFloat viewY = normalizedY * viewHeight;
        
        // Set color based on label (0 = red, 1 = blue)
        NSColor* color;
        if (label == 0) {
            color = [NSColor redColor];
        } else if (label == 1) {
            color = [NSColor blueColor];
        } else {
            color = [NSColor grayColor]; // fallback for unexpected labels
        }
        
        [color setFill];
        
        // Draw disc
        NSRect discRect = NSMakeRect(viewX - discRadius, 
                                     viewY - discRadius,
                                     discRadius * 2, 
                                     discRadius * 2);
        NSBezierPath* circle = [NSBezierPath bezierPathWithOvalInRect:discRect];
        [circle fill];
    }
    
    // Draw border
    [[NSColor blackColor] setStroke];
    [NSBezierPath setDefaultLineWidth:1.0];
    [NSBezierPath strokeRect:bounds];
}

@end
