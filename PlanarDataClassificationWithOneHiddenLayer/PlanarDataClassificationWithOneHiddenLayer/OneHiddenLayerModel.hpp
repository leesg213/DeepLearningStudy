//
//  OneHiddenLayerModel.hpp
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#ifndef OneHiddenLayerModel_hpp
#define OneHiddenLayerModel_hpp

#include <stdio.h>
#include <vector>
#include <random>
#include <Metal/Metal.h>
#include <simd/simd.h>

class OneHiddenLayerModel
{
public:
    void Init();
    
    void TrainF(std::vector<float> const& train_x,
               std::vector<uint8_t> const& train_y,
                int num_hidden_layers,
               int num_trains,
               int num_features,
               int numIterations,
               float learningRate);
    
    void PredictF(std::vector<float> const& test_x,
                  int num_hidden_layers,
                  int num_tests,
                  int num_features,
                 std::vector<float>& out_results);
    
    // Port of np.random.randn() - generates random numbers from standard normal distribution
    // Returns a single random value
    float randn();
    
    // Port of np.random.randn(d0, d1, ...) - generates array of random values
    // Returns a vector of random values from standard normal distribution
    std::vector<float> randn(size_t size);
    
    // Set random seed for reproducibility (equivalent to np.random.seed())
    void seed(unsigned int seedValue);
    
private:
    id<MTLDevice> _device;
    id<MTLCommandQueue> _cmdQueue;
    
    id<MTLComputePipelineState> _pipelineComputeCostF_1HiddenLayer, _pipelineComputeGradsF_1HiddenLayer;
    
    std::vector<float> _weights;
    
    float CalcCost(id<MTLBuffer> allCosts, size_t num_trains);
    
    // Random number generator for randn()
    std::mt19937 _rng;
    std::normal_distribution<float> _normalDist;
    bool _rngInitialized;
};


#endif /* OneHiddenLayerModel_hpp */
