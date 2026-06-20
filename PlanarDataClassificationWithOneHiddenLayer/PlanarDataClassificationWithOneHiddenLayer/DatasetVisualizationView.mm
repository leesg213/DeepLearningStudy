//
//  DatasetVisualizationView.mm
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#import "DatasetVisualizationView.h"
#include <vector>
#include <algorithm>

@implementation DatasetVisualizationView {
    std::vector<std::vector<double>> _X;
    std::vector<std::vector<uint8_t>> _Y;
    double _minX, _maxX, _minY, _maxY;
}

- (void)setDatasetX:(const std::vector<std::vector<double>>&)X 
                  Y:(const std::vector<std::vector<uint8_t>>&)Y {
    _X = X;
    _Y = Y;
    
    // Calculate bounds for scaling
    if (!X.empty() && !X[0].empty()) {
        _minX = _maxX = X[0][0];
        _minY = _maxY = X[1][0];
        
        for (size_t i = 0; i < X[0].size(); i++) {
            _minX = std::min(_minX, X[0][i]);
            _maxX = std::max(_maxX, X[0][i]);
            _minY = std::min(_minY, X[1][i]);
            _maxY = std::max(_maxY, X[1][i]);
        }
        
        // Add some padding
        double paddingX = (_maxX - _minX) * 0.1;
        double paddingY = (_maxY - _minY) * 0.1;
        _minX -= paddingX;
        _maxX += paddingX;
        _minY -= paddingY;
        _maxY += paddingY;
    }
    
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];
    
    // Fill background with white
    [[NSColor whiteColor] setFill];
    NSRectFill(dirtyRect);
    
    if (_X.empty() || _Y.empty() || _X[0].empty() || _Y[0].empty()) {
        return;
    }
    
    NSRect bounds = [self bounds];
    double width = NSWidth(bounds);
    double height = NSHeight(bounds);
    
    // Calculate scale to fit data in view
    double scaleX = width / (_maxX - _minX);
    double scaleY = height / (_maxY - _minY);
    
    // Use the smaller scale to maintain aspect ratio
    double scale = std::min(scaleX, scaleY);
    
    // Calculate offsets to center the data
    double offsetX = (width - (_maxX - _minX) * scale) / 2.0;
    double offsetY = (height - (_maxY - _minY) * scale) / 2.0;
    
    // Disc radius in points
    double discRadius = 3.0;
    
    // Draw each data point
    for (size_t i = 0; i < _X[0].size(); i++) {
        double x = _X[0][i];
        double y = _X[1][i];
        uint8_t label = _Y[0][i];
        
        // Transform to view coordinates
        double viewX = (x - _minX) * scale + offsetX;
        double viewY = (y - _minY) * scale + offsetY;
        
        // Set color based on label
        // Y=0 -> Blue, Y=1 -> Red
        NSColor *color = (label == 1) ? [NSColor blueColor] : [NSColor redColor];
        [color setFill];
        
        // Draw disc
        NSRect discRect = NSMakeRect(viewX - discRadius, 
                                     viewY - discRadius,
                                     discRadius * 2, 
                                     discRadius * 2);
        NSBezierPath *path = [NSBezierPath bezierPathWithOvalInRect:discRect];
        [path fill];
    }
}

@end
