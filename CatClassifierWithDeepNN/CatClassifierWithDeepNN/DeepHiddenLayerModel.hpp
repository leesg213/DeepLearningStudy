//
//  DeepHiddenLayerModel.hpp
//  PlanarDataClassificationWithDeepHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#ifndef DeepHiddenLayerModel_hpp
#define DeepHiddenLayerModel_hpp

#include <stdio.h>
#include <vector>
#include <random>
#include <Metal/Metal.h>
#include <simd/simd.h>

class DeepHiddenLayerModel
{
public:
    void Init(std::vector<int> const& inLayers);
    
    void TrainF(std::vector<float> const& train_x,
               std::vector<uint8_t> const& train_y,
               int num_trains,
               int numIterations,
               float learningRate);
    
    void PredictF(std::vector<float> const& test_x,
                  int num_tests,
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
    
    id<MTLComputePipelineState> _pipelineDeepForwardRelu, _pipelineDeepForwardSigmoid, _pipelineDeepComputeDAL, _pipelineDeepComputeGradSigmoid, _pipelineDeepComputeGradRelu, _pipelineOptimize, _pipelineClearGrads;
    
    std::vector<float> _weights;
    
    float ComputeCost(id<MTLBuffer> activation, size_t activationOffset, std::vector<uint8_t> const& label);
    
    std::vector<int> Layers;
    
    // Random number generator for randn()
    std::mt19937 _rng;
    std::normal_distribution<float> _normalDist;
    bool _rngInitialized;
};


#endif /* DeepHiddenLayerModel_hpp */
