//
//  CatClassifier.hpp
//  LogisticRegression00
//
//  Created by Sugu Lee on 6/19/26.
//

#ifndef CatClassifier_hpp
#define CatClassifier_hpp

#include <stdio.h>
#include <vector>

#import <Metal/Metal.h>
#import <simd/simd.h>

class CatClassifier
{
public:
    
    void Init(id<MTLDevice> inDevice);
    void Train(std::vector<std::vector<uint8_t>> const& trainData,
               std::vector<uint8_t> const& isCatData,
               int numIterations,
               float learningRate);
    
private:
    id<MTLDevice> _device;
    id<MTLCommandQueue> _cmdQueue;
    id<MTLBuffer> _weights;
    id<MTLComputePipelineState> _pipelineComputeCosts;
    
    void Forward(std::vector<uint8_t> const& trainData, bool isCat);
    
};

#endif /* CatClassifier_hpp */
