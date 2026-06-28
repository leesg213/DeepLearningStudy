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
#include <functional>
#include <Metal/Metal.h>
#include <simd/simd.h>
#include "NumpyRandn.h"

class DeepHiddenLayerModel
{
public:
    void Init(std::vector<int> const& inLayers);
    
    typedef std::function<void(int iteration, float cost)> CostCallback;
    
    void TrainF(std::vector<float> const& train_x,
               std::vector<uint8_t> const& train_y,
               int numIterations,
               float learningRate,
               int logInterval = 100,
               std::vector<std::pair<int, float>>* out_costs = nullptr,
               CostCallback costCallback = nullptr);
    
    void PredictF(std::vector<float> const& test_x,
                  int num_tests,
                 std::vector<float>& out_results);
    
private:
    id<MTLDevice> _device;
    id<MTLCommandQueue> _cmdQueue;
    
    id<MTLComputePipelineState> _pipelineDeepForward, _pipelineDeepComputeDAL, _pipelineDeepComputeGrad, _pipelineOptimize, _pipelineClearGrads;
    
    std::vector<float> _weights;
    
    float ComputeCost(id<MTLBuffer> activation, size_t activationOffset, std::vector<uint8_t> const& label);
    
    std::vector<int> Layers;
    
    NumpyRandn Random;
};


#endif /* DeepHiddenLayerModel_hpp */
