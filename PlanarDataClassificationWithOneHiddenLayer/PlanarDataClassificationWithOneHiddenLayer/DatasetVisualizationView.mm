//
//  DatasetVisualizationView.mm
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#import "DatasetVisualizationView.h"
#include <vector>
#include <algorithm>
#include "LogisticRegressionModel.hpp"
#include "OneHiddenLayerModel.hpp"

@implementation DatasetVisualizationView {
    std::vector<std::vector<float>> _X;
    std::vector<std::vector<uint8_t>> _Y;
    float _minX, _maxX, _minY, _maxY;
}

- (void)setDatasetX:(const std::vector<std::vector<float>>&)X
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

- (void)drawLogisticRegressionResult
{
    LogisticRegressionModel model;
    model.Init();
    
    int num_features = _X.size();
    int num_trains = _X[0].size();
    
    std::vector<float> train_x(num_trains * num_features);
    for(int i = 0;i<num_trains;++i)
    {
        train_x[i*num_features+0] = _X[0][i] / _maxX;
        train_x[i*num_features+1] = _X[1][i] / _maxY;
        
    }
    std::vector<uint8_t> train_y(num_trains);
    memcpy(train_y.data(), _Y[0].data(), num_trains);
    
    model.TrainF(train_x, train_y, num_trains, num_features, 2000, 0.1);
    
    int num_plots = 100;
    float x_increment = (_maxX - _minX) / num_plots;
    float y_increment = (_maxY - _minY) / num_plots;
    
    std::vector<float> test_x;
    
    for(float x = _minX;x<_maxX;x += x_increment)
    {
        for(float y= _minY;y<_maxY; y += y_increment)
        {
            test_x.push_back(x/_maxX);
            test_x.push_back(y/_maxY);
        }
    }
    
    int num_tests = test_x.size()/2;
    std::vector<float> predict_results;

    model.PredictF(test_x, num_tests, 2, predict_results);
    
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
    
    for(int i = 0;i<num_tests;++i)
    {
        float result = predict_results[i];
        
        double x = test_x[i*2+0]*_maxX;
        double y = test_x[i*2+1]*_maxY;
        uint8_t label = predict_results[i]>0.5 ? 1 : 0;
        
        // Transform to view coordinates
        double viewX = (x - _minX) * scale + offsetX;
        double viewY = (y - _minY) * scale + offsetY;
        
        // Set color based on label
        // Y=0 -> Blue, Y=1 -> Red
        NSColor *color = (label == 1) ? [NSColor cyanColor] : [NSColor systemPinkColor];
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


- (void)drawOneHiddenLayerResult
{
    OneHiddenLayerModel model;
    model.Init();
    
    int num_features = _X.size();
    int num_trains = _X[0].size();
    
    std::vector<float> train_x(num_trains * num_features);
    for(int i = 0;i<num_trains;++i)
    {
        train_x[i*num_features+0] = _X[0][i] / _maxX;
        train_x[i*num_features+1] = _X[1][i] / _maxY;
        
    }
    std::vector<uint8_t> train_y(num_trains);
    memcpy(train_y.data(), _Y[0].data(), num_trains);
    
    model.TrainF(train_x, train_y, 4, num_trains, num_features, 10000, 1.2);
    
#if true
    int num_plots = 100;
    float x_increment = (_maxX - _minX) / num_plots;
    float y_increment = (_maxY - _minY) / num_plots;
    
    std::vector<float> test_x;
    
    for(float x = _minX;x<_maxX;x += x_increment)
    {
        for(float y= _minY;y<_maxY; y += y_increment)
        {
            test_x.push_back(x/_maxX);
            test_x.push_back(y/_maxY);
        }
    }
    
    int num_tests = test_x.size()/2;
    std::vector<float> predict_results;

    model.PredictF(test_x, 4, num_tests, 2, predict_results);
    
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
    
    for(int i = 0;i<num_tests;++i)
    {
        float result = predict_results[i];
        
        double x = test_x[i*2+0]*_maxX;
        double y = test_x[i*2+1]*_maxY;
        uint8_t label = predict_results[i]>0.5 ? 1 : 0;
        
        // Transform to view coordinates
        double viewX = (x - _minX) * scale + offsetX;
        double viewY = (y - _minY) * scale + offsetY;
        
        // Set color based on label
        // Y=0 -> Blue, Y=1 -> Red
        NSColor *color = (label == 1) ? [NSColor cyanColor] : [NSColor systemPinkColor];
        [color setFill];
        
        // Draw disc
        NSRect discRect = NSMakeRect(viewX - discRadius,
                                     viewY - discRadius,
                                     discRadius * 2,
                                     discRadius * 2);
        NSBezierPath *path = [NSBezierPath bezierPathWithOvalInRect:discRect];
        [path fill];
    }
#endif
}
- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];
    
    // Fill background with white
    [[NSColor whiteColor] setFill];
    NSRectFill(dirtyRect);
    
    if (_X.empty() || _Y.empty() || _X[0].empty() || _Y[0].empty()) {
        return;
    }
    
    //[self drawLogisticRegressionResult];
    [self drawOneHiddenLayerResult];
    
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
