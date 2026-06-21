//
//  DatasetVisualizationView.h
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#import <Cocoa/Cocoa.h>
#include <vector>

@interface DatasetVisualizationView : NSView

// Set the dataset to visualize
// X: 2 x m matrix (features x examples)
// Y: 1 x m matrix (labels x examples)
- (void)setDatasetX:(const std::vector<std::vector<float>>&)X 
                  Y:(const std::vector<std::vector<uint8_t>>&)Y;

@end
