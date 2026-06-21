//
//  LogisticRegressionModel.hpp
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#ifndef LogisticRegressionModel_hpp
#define LogisticRegressionModel_hpp

#include <stdio.h>
#include <vector>
#include <Metal/Metal.h>

class LogisticRegressionModel
{
    
public:
    void Init();
    void Train(std::vector<uint8_t> const& train_x,
               std::vector<uint8_t> const& train_y,
               int num_trains,
               int num_features,
               int numIterations,
               float learningRate);
    
    int Predict(std::vector<uint8_t> const& test_x,
                  std::vector<uint8_t> const& test_y,
                  int num_tests,
                  int num_features);
    
    void TrainF(std::vector<float> const& train_x,
               std::vector<uint8_t> const& train_y,
               int num_trains,
               int num_features,
               int numIterations,
               float learningRate);
    
    void PredictF(std::vector<float> const& test_x,
                  int num_tests,
                  int num_features,
                 std::vector<float>& out_results);
private:
    float CalcCost(id<MTLBuffer> allCosts, size_t num_trains);
    
    id<MTLDevice> _device;
    id<MTLCommandQueue> _cmdQueue;
    id<MTLBuffer> _weights;
    id<MTLComputePipelineState> _pipelineComputeCosts, _pipelineComputeGrads;
    id<MTLComputePipelineState> _pipelineComputeCostsF, _pipelineComputeGradsF;
    std::vector<float> trained_weights;
};

#endif /* LogisticRegressionModel_hpp */
