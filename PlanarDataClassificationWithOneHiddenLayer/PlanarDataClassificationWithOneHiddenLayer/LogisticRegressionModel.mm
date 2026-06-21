//
//  LogisticRegressionModel.cpp
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#include "LogisticRegressionModel.hpp"
#include "ShaderTypes.h"
#include <simd/simd.h>

static int capture_trace_iter = -1;

void LogisticRegressionModel::Init()
{
    _device = MTLCreateSystemDefaultDevice();
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
    _pipelineComputeCostsF = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"computeCostF"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    _pipelineComputeGradsF = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"computeGradsF"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
}

float LogisticRegressionModel::CalcCost(id<MTLBuffer> allCosts, size_t num_trains)
{
    float sum = 0;
    float* allCostValues = (float*)allCosts.contents;
    for(size_t i = 0;i<num_trains;++i)
    {
        float costValue = allCostValues[i];
        sum += allCostValues[i];
    }
    
    float cost = -sum/num_trains;
    return cost;
}

float sigmoid(float inValue)
{
    float epsilon = 0.000001f;
    return simd_clamp((1.0f / (1 + exp(-inValue))),epsilon,1-epsilon);
}

int LogisticRegressionModel::Predict(std::vector<uint8_t> const& test_x,
                                      std::vector<uint8_t> const& test_y,
                                      int num_tests,
                                      int num_features)
{
    int num_passed = 0;
    float* weight_values = (float*)_weights.contents;

    for(int i = 0;i<num_tests;++i)
    {
        float sum = 0;
        for(int j = 0;j<num_features;++j)
        {
            sum += weight_values[j+1] * (test_x[i*num_features+j]/255.0f);
        }
        sum += weight_values[0];
        
        float predict = sigmoid(sum);
        
        bool predict_result = predict > 0.5f;
        
        if(predict_result == test_y[i])
        {
            num_passed++;
        }
    }
        
    NSLog(@"%d/%d passed.", num_passed, num_tests);
    
    return num_passed;
}

void LogisticRegressionModel::Train(std::vector<uint8_t> const& train_x,
                                    std::vector<uint8_t> const& train_y,
                                    int num_trains,
                                    int num_features,
                                    int numIterations,
                                    float learningRate)
{
    
    size_t numFeatures = num_features;
    
    id<MTLBuffer> uniforms = [_device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    Uniforms* uniforms_data = (Uniforms*)uniforms.contents;
    uniforms_data->numImages = num_trains;
    uniforms_data->numFeatures = numFeatures;
    uniforms_data->normalizer_scaler = 1;
    
    id<MTLBuffer> outCosts = [_device newBufferWithLength:num_trains*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outActivations = [_device newBufferWithLength:num_trains*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outGrads = [_device newBufferWithLength:(num_features+1)*sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLBuffer> trainingDataBuffer = [_device newBufferWithLength:train_x.size() options:(MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared)];
    memcpy(trainingDataBuffer.contents, train_x.data(), train_x.size());
    
    id<MTLBuffer> isCatDataBuffer = [_device newBufferWithLength:num_trains options:MTLResourceStorageModeShared];
    memcpy(isCatDataBuffer.contents, &train_y[0], num_trains);
    
    _weights = [_device newBufferWithLength:(numFeatures+1)*sizeof(float) options:MTLResourceStorageModeShared];
    memset(_weights.contents, 0, _weights.length);
    
    for(int iter = 0;iter < numIterations; ++iter)
    {
        NSLog(@"Iteration : %d", iter);
        
        // Start Metal GPU capture
        MTLCaptureManager *captureManager = nil;
        if(iter == capture_trace_iter)
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
        [encoder dispatchThreads:MTLSizeMake(num_trains, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        
       // [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        
        [encoder setComputePipelineState:_pipelineComputeGrads];
        [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
        [encoder setBuffer:isCatDataBuffer offset:0 atIndex: 1];
        [encoder setBuffer:outActivations offset:0 atIndex: 2];
        [encoder setBuffer:outGrads offset:0 atIndex: 3];
        [encoder setBuffer:uniforms offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(num_features+1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        
        float cost = CalcCost(outCosts, num_trains);
        NSLog(@"Cost : %f", cost);
        
        float* gradValues = (float*)outGrads.contents;
        // Update weights
        float* weight_values = (float*)_weights.contents;
        for(int i = 0;i<numFeatures+1;++i)
        {
            weight_values[i] = weight_values[i] - learningRate * gradValues[i];
        }
        
        if(captureManager)
        {
            // Stop Metal GPU capture
            [captureManager stopCapture];
            NSLog(@"Metal GPU capture stopped");
        }
    }
    
    {
        float* weight_values = (float*)_weights.contents;
        trained_weights.resize(numFeatures+1);
        for(int i = 0;i<numFeatures+1;++i)
        {
            trained_weights[i] = weight_values[i];
        }
    }
    
    NSLog(@"Train complete");
}


void LogisticRegressionModel::PredictF(std::vector<float> const& test_x,
                                      int num_tests,
                                      int num_features,
                                      std::vector<float>& out_results)
{
    
    out_results.clear();
    
    int num_passed = 0;
    float* weight_values = (float*)_weights.contents;

    for(int i = 0;i<num_tests;++i)
    {
        float sum = 0;
        for(int j = 0;j<num_features;++j)
        {
            sum += weight_values[j+1] * (test_x[i*num_features+j]);
        }
        sum += weight_values[0];
        
        float predict = sigmoid(sum);
        
        out_results.push_back(predict);
    }
}

void LogisticRegressionModel::TrainF(std::vector<float> const& train_x,
                                    std::vector<uint8_t> const& train_y,
                                    int num_trains,
                                    int num_features,
                                    int numIterations,
                                    float learningRate)
{
    
    size_t numFeatures = num_features;
    
    id<MTLBuffer> uniforms = [_device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    Uniforms* uniforms_data = (Uniforms*)uniforms.contents;
    uniforms_data->numImages = num_trains;
    uniforms_data->numFeatures = numFeatures;
    uniforms_data->normalizer_scaler = 1;
    
    id<MTLBuffer> outCosts = [_device newBufferWithLength:num_trains*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outActivations = [_device newBufferWithLength:num_trains*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outGrads = [_device newBufferWithLength:(num_features+1)*sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLBuffer> trainingDataBuffer = [_device newBufferWithLength:train_x.size()*sizeof(float) options:(MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared)];
    memcpy(trainingDataBuffer.contents, train_x.data(), train_x.size()*sizeof(float));
    
    id<MTLBuffer> isCatDataBuffer = [_device newBufferWithLength:num_trains options:MTLResourceStorageModeShared];
    memcpy(isCatDataBuffer.contents, &train_y[0], num_trains);
    
    _weights = [_device newBufferWithLength:(numFeatures+1)*sizeof(float) options:MTLResourceStorageModeShared];
    memset(_weights.contents, 0, _weights.length);
    
    for(int iter = 0;iter < numIterations; ++iter)
    {
        NSLog(@"Iteration : %d", iter);
        
        // Start Metal GPU capture
        MTLCaptureManager *captureManager = nil;
        if(iter == capture_trace_iter)
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
        
        [encoder setComputePipelineState:_pipelineComputeCostsF];
        [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
        [encoder setBuffer:isCatDataBuffer offset:0 atIndex: 1];
        [encoder setBuffer:_weights offset:0 atIndex:2];
        [encoder setBuffer:uniforms offset:0 atIndex:3];
        [encoder setBuffer:outCosts offset:0 atIndex:4];
        [encoder setBuffer:outActivations offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(num_trains, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        
       // [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        
        [encoder setComputePipelineState:_pipelineComputeGradsF];
        [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
        [encoder setBuffer:isCatDataBuffer offset:0 atIndex: 1];
        [encoder setBuffer:outActivations offset:0 atIndex: 2];
        [encoder setBuffer:outGrads offset:0 atIndex: 3];
        [encoder setBuffer:uniforms offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(num_features+1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        
        float cost = CalcCost(outCosts, num_trains);
        NSLog(@"Cost : %f", cost);
        
        float* gradValues = (float*)outGrads.contents;
        // Update weights
        float* weight_values = (float*)_weights.contents;
        for(int i = 0;i<numFeatures+1;++i)
        {
            weight_values[i] = weight_values[i] - learningRate * gradValues[i];
        }
        
        if(captureManager)
        {
            // Stop Metal GPU capture
            [captureManager stopCapture];
            NSLog(@"Metal GPU capture stopped");
        }
    }
    
    {
        float* weight_values = (float*)_weights.contents;
        trained_weights.resize(numFeatures+1);
        for(int i = 0;i<numFeatures+1;++i)
        {
            trained_weights[i] = weight_values[i];
        }
    }
    
    NSLog(@"Train complete");
}
