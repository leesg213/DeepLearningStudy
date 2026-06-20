//
//  ViewController.h
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#import <Cocoa/Cocoa.h>
#include <vector>
#include <utility>

@interface ViewController : NSViewController

// C++ method to generate a planar dataset for classification
// Returns a pair of (X, Y) where:
// - X is a 2 x 400 matrix of features (transposed)
// - Y is a 1 x 400 matrix of labels (transposed)
- (std::pair<std::vector<std::vector<double>>, std::vector<std::vector<uint8_t>>>)loadPlanarDataset;

// Visualize the planar dataset
- (void)visualizeDataset;

@end

