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

// Helper struct for coordinate transformation
struct ViewTransform {
    double scale;
    double offsetX;
    double offsetY;
    double width;
    double height;
};

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

#pragma mark - Helper Methods

// Calculate view transform for mapping data coordinates to view coordinates
- (ViewTransform)calculateViewTransform {
    NSRect bounds = [self bounds];
    double width = NSWidth(bounds);
    double height = NSHeight(bounds);
    
    ViewTransform transform;
    transform.width = width;
    transform.height = height;
    
    // Calculate scale to fit data in view
    double scaleX = width / (_maxX - _minX);
    double scaleY = height / (_maxY - _minY);
    
    // Use the smaller scale to maintain aspect ratio
    transform.scale = std::min(scaleX, scaleY);
    
    // Calculate offsets to center the data
    transform.offsetX = (width - (_maxX - _minX) * transform.scale) / 2.0;
    transform.offsetY = (height - (_maxY - _minY) * transform.scale) / 2.0;
    
    return transform;
}

// Prepare normalized training data from the dataset
- (std::vector<float>)prepareTrainingData:(int*)outNumTrains numFeatures:(int*)outNumFeatures {
    int num_features = _X.size();
    int num_trains = _X[0].size();
    
    if (outNumTrains) *outNumTrains = num_trains;
    if (outNumFeatures) *outNumFeatures = num_features;
    
    std::vector<float> train_x(num_trains * num_features);
    for (int i = 0; i < num_trains; ++i) {
        train_x[i * num_features + 0] = _X[0][i] / _maxX;
        train_x[i * num_features + 1] = _X[1][i] / _maxY;
    }
    
    return train_x;
}

// Prepare training labels
- (std::vector<uint8_t>)prepareTrainingLabels:(int)numTrains {
    std::vector<uint8_t> train_y(numTrains);
    memcpy(train_y.data(), _Y[0].data(), numTrains);
    return train_y;
}

// Generate test points for prediction visualization
- (std::vector<float>)generateTestPoints:(int)numPlotsPerAxis numTests:(int*)outNumTests {
    float x_increment = (_maxX - _minX) / numPlotsPerAxis;
    float y_increment = (_maxY - _minY) / numPlotsPerAxis;
    
    std::vector<float> test_x;
    
    for (float x = _minX; x < _maxX; x += x_increment) {
        for (float y = _minY; y < _maxY; y += y_increment) {
            test_x.push_back(x / _maxX);
            test_x.push_back(y / _maxY);
        }
    }
    
    if (outNumTests) *outNumTests = test_x.size() / 2;
    return test_x;
}

// Draw prediction results as colored discs
- (void)drawPredictionResults:(const std::vector<float>&)testPoints
                   predictions:(const std::vector<float>&)predictions
                     transform:(ViewTransform)transform
                    discRadius:(double)discRadius {
    
    int numTests = testPoints.size() / 2;
    
    for (int i = 0; i < numTests; ++i) {
        double x = testPoints[i * 2 + 0] * _maxX;
        double y = testPoints[i * 2 + 1] * _maxY;
        uint8_t label = predictions[i] > 0.5 ? 1 : 0;
        
        // Transform to view coordinates
        double viewX = (x - _minX) * transform.scale + transform.offsetX;
        double viewY = (y - _minY) * transform.scale + transform.offsetY;
        
        // Set color based on label
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

#pragma mark - Model Training and Prediction

- (void)drawLogisticRegressionResult
{
    LogisticRegressionModel model;
    model.Init();
    
    int num_trains, num_features;
    std::vector<float> train_x = [self prepareTrainingData:&num_trains numFeatures:&num_features];
    std::vector<uint8_t> train_y = [self prepareTrainingLabels:num_trains];
    
    model.TrainF(train_x, train_y, num_trains, num_features, 2000, 0.1);
    
    // Generate test points
    int num_tests;
    std::vector<float> test_x = [self generateTestPoints:100 numTests:&num_tests];
    
    // Get predictions
    std::vector<float> predict_results;
    model.PredictF(test_x, num_tests, 2, predict_results);
    
    // Draw predictions
    ViewTransform transform = [self calculateViewTransform];
    [self drawPredictionResults:test_x predictions:predict_results transform:transform discRadius:3.0];
}


- (void)drawOneHiddenLayerResult
{
    OneHiddenLayerModel model;
    model.Init();
    
    int num_trains, num_features;
    std::vector<float> train_x = [self prepareTrainingData:&num_trains numFeatures:&num_features];
    std::vector<uint8_t> train_y = [self prepareTrainingLabels:num_trains];
    
    int hidden_layer_size = 4;
    
    model.TrainF(train_x, train_y, hidden_layer_size, num_trains, num_features, 10000, 1.2);
    
#if true
    // Generate test points
    int num_tests;
    std::vector<float> test_x = [self generateTestPoints:100 numTests:&num_tests];
    
    // Get predictions
    std::vector<float> predict_results;
    model.PredictF(test_x, hidden_layer_size, num_tests, 2, predict_results);
    
    // Draw predictions
    ViewTransform transform = [self calculateViewTransform];
    [self drawPredictionResults:test_x predictions:predict_results transform:transform discRadius:3.0];
#endif
}

#pragma mark - Drawing
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
    
    // Calculate view transform once
    ViewTransform transform = [self calculateViewTransform];
    
    // Disc radius in points
    double discRadius = 3.0;
    
    // Draw each training data point
    for (size_t i = 0; i < _X[0].size(); i++) {
        double x = _X[0][i];
        double y = _X[1][i];
        uint8_t label = _Y[0][i];
        
        // Transform to view coordinates
        double viewX = (x - _minX) * transform.scale + transform.offsetX;
        double viewY = (y - _minY) * transform.scale + transform.offsetY;
        
        // Set color based on label
        // Y=0 -> Red, Y=1 -> Blue
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
