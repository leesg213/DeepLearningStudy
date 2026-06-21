//
//  ViewController.mm
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#import "ViewController.h"
#import "DatasetVisualizationView.h"
#include <vector>
#include <random>
#include <cmath>

@implementation ViewController {
    std::vector<std::vector<float>> _datasetX;
    std::vector<std::vector<uint8_t>> _datasetY;
    DatasetVisualizationView *_visualizationView;
}

- (void)viewDidLoad {
    [super viewDidLoad];

    // Do any additional setup after loading the view.
    
    auto dataset = [self loadPlanarDataset];
    _datasetX = dataset.first;
    _datasetY = dataset.second;
    
    [self visualizeDataset];
}


- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}

#pragma mark - Visualization

- (void)visualizeDataset {
    // Create visualization view if it doesn't exist
    if (!_visualizationView) {
        NSRect frame = self.view.bounds;
        _visualizationView = [[DatasetVisualizationView alloc] initWithFrame:frame];
        _visualizationView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        [self.view addSubview:_visualizationView];
    }
    
    // Set the dataset to visualize
    [_visualizationView setDatasetX:_datasetX Y:_datasetY];
}

#pragma mark - C++ Methods

- (std::pair<std::vector<std::vector<float>>, std::vector<std::vector<uint8_t>>>)loadPlanarDataset {
    // Set random seed for reproducibility
    std::mt19937 rng(1);
    std::normal_distribution<float> normal_dist(0.0, 1.0);
    
    const int m = 400; // number of examples
    const int N = m / 2; // number of points per class
    const int D = 2; // dimensionality
    const float a = 4.0; // maximum ray of the flower
    
    // Initialize X matrix (m x D) and Y labels (m x 1)
    std::vector<std::vector<float>> X(m, std::vector<float>(D, 0.0));
    std::vector<std::vector<uint8_t>> Y(m, std::vector<uint8_t>(1, 0));
    
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < N; i++) {
            int idx = N * j + i;
            
            // Generate theta: linspace from j*3.12 to (j+1)*3.12 with random noise
            float t = j * 3.12 + (float)i / (N - 1) * 3.12 + normal_dist(rng) * 0.2;
            
            // Generate radius: a*sin(4*t) with random noise
            float r = a * std::sin(4.0 * t) + normal_dist(rng) * 0.2;
            
            // Set X values: [r*sin(t), r*cos(t)]
            X[idx][0] = r * std::sin(t);
            X[idx][1] = r * std::cos(t);
            
            // Set Y label
            Y[idx][0] = j;
        }
    }
    
    // Transpose X (m x D -> D x m)
    std::vector<std::vector<float>> X_T(D, std::vector<float>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < D; j++) {
            X_T[j][i] = X[i][j];
        }
    }
    
    // Transpose Y (m x 1 -> 1 x m)
    std::vector<std::vector<uint8_t>> Y_T(1, std::vector<uint8_t>(m));
    for (int i = 0; i < m; i++) {
        Y_T[0][i] = Y[i][0];
    }
    
    return std::make_pair(X_T, Y_T);
}

@end
