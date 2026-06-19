//
//  CatClassifier.cpp
//  LogisticRegression00
//
//  Created by Sugu Lee on 6/19/26.
//

#include "CatClassifier.hpp"
#include "ShaderTypes.h"

static bool captureTrace = false;

void CatClassifier::Init(id<MTLDevice> inDevice)
{
    _device = inDevice;
    _cmdQueue = [_device newCommandQueue];
    
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
    
    NSError* error = nil;
    
    _pipelineComputeCosts = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"computeCost"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    _pipelineComputeGrads = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"computeGrads"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
}

float CatClassifier::CalcCost(id<MTLBuffer> allCosts, size_t numImages)
{
    float sum = 0;
    float* allCostValues = (float*)allCosts.contents;
    for(size_t i = 0;i<numImages;++i)
    {
        float costValue = allCostValues[i];
        if(isnan(costValue))
        {
            NSLog(@"here");
        }
        sum += allCostValues[i];
    }
    
    float cost = -sum/numImages;
    
    if(isnan(cost))
    {
        NSLog(@"Here");
    }
    return cost;
}
void CatClassifier::CalcGrad(std::vector<std::vector<uint8_t>> const& trainData, id<MTLBuffer> allActivations, std::vector<uint8_t> const& isCatData, std::vector<float>& outDw, float& outDb)
{
    size_t numImages = trainData.size();
    size_t imageDataLength = trainData[0].size();
    
    std::vector<float> AminusY;
    AminusY.resize(numImages);
    
    float* allActivationsValues = (float*)allActivations.contents;
    for(int i = 0;i<numImages;++i)
    {
        AminusY[i] = allActivationsValues[i] - isCatData[i];
    }
    
    outDw.resize(imageDataLength);
    
    for(int i = 0;i<imageDataLength;++i)
    {
        float dotProduct = 0;
        for(int j = 0;j<numImages;++j)
        {
            dotProduct += trainData[j][i]/255.0f * AminusY[j];
        }
        dotProduct /= numImages;
        outDw[i] = dotProduct;
    }
    
    float sum = 0;
    for(int i = 0;i<numImages;++i)
    {
        sum += AminusY[i];
    }
    outDb = sum / numImages;
}
void CatClassifier::Train(std::vector<std::vector<uint8_t>> const& trainData,
                          std::vector<uint8_t> const& isCatData,
                          int numIterations,
                          float learningRate)
{
    
    size_t numImages = trainData.size();
    size_t imageDataLength = trainData[0].size();
    
    id<MTLBuffer> uniforms = [_device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    Uniforms* uniforms_data = (Uniforms*)uniforms.contents;
    uniforms_data->numImages = numImages;
    uniforms_data->imageDataLength = imageDataLength;
    
    id<MTLBuffer> outCosts = [_device newBufferWithLength:numImages*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outActivations = [_device newBufferWithLength:numImages*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outGrads = [_device newBufferWithLength:(imageDataLength+1)*sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLBuffer> trainingDataBuffer = [_device newBufferWithLength:numImages*imageDataLength options:(MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared)];
    memcpy(trainingDataBuffer.contents, &trainData[0],numImages*imageDataLength);
    
    id<MTLBuffer> isCatDataBuffer = [_device newBufferWithLength:numImages options:MTLResourceStorageModeShared];
    memcpy(isCatDataBuffer.contents, &isCatData[0], numImages);
    
    _weights = [_device newBufferWithLength:(imageDataLength+1)*sizeof(float) options:MTLResourceStorageModeShared];
    memset(_weights.contents, 0, _weights.length);
    
    for(int iter = 0;iter < numIterations; ++iter)
    {
        NSLog(@"Iteration : %d", iter);
        
        // Start Metal GPU capture
        MTLCaptureManager *captureManager = nil;
        if(captureTrace && iter == 6)
        {
            captureManager = [MTLCaptureManager sharedCaptureManager];
            MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
            captureDescriptor.captureObject = _device;
            captureDescriptor.destination = MTLCaptureDestinationDeveloperTools;
            
            NSError *captureError = nil;
            if ([captureManager startCaptureWithDescriptor:captureDescriptor error:&captureError]) {
                NSLog(@"Metal GPU capture started");
            } else {
                NSLog(@"Failed to start Metal GPU capture: %@", captureError.localizedDescription);
                
            }
        }
        
        id<MTLCommandBuffer> cmdBuffer = [_cmdQueue commandBuffer];
        cmdBuffer.label = @"Training Command Buffer";
        
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        encoder.label = @"Training Compute Encoder";
        
        [encoder setComputePipelineState:_pipelineComputeCosts];
        [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
        [encoder setBuffer:isCatDataBuffer offset:0 atIndex: 1];
        [encoder setBuffer:_weights offset:0 atIndex:2];
        [encoder setBuffer:uniforms offset:0 atIndex:3];
        [encoder setBuffer:outCosts offset:0 atIndex:4];
        [encoder setBuffer:outActivations offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(numImages, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        
        [encoder setComputePipelineState:_pipelineComputeGrads];
        [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
        [encoder setBuffer:isCatDataBuffer offset:0 atIndex: 1];
        [encoder setBuffer:outActivations offset:0 atIndex: 2];
        [encoder setBuffer:outGrads offset:0 atIndex: 3];
        [encoder setBuffer:uniforms offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(imageDataLength+1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        
        float cost = CalcCost(outCosts, numImages);
        std::vector<float> dW;
        float dB;
        //CalcGrad(trainData, outActivations, isCatData, dW, dB);
        NSLog(@"Cost : %f", cost);
        
        float* gradValues = (float*)outGrads.contents;
        // Update weights
        float* weight_values = (float*)_weights.contents;
        for(int i = 0;i<dW.size();++i)
        {
            weight_values[i+1] = weight_values[i+1] - learningRate * gradValues[i+1];
        }
        weight_values[0] = weight_values[0] - learningRate*gradValues[0];
        
        if(captureManager)
        {
            // Stop Metal GPU capture
            [captureManager stopCapture];
            NSLog(@"Metal GPU capture stopped");
        }
    }
}
