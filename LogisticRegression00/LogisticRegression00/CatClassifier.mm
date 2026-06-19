//
//  CatClassifier.cpp
//  LogisticRegression00
//
//  Created by Sugu Lee on 6/19/26.
//

#include "CatClassifier.hpp"
#include "ShaderTypes.h"

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
    uniforms_data->imageDataLength = imageDataLength;
    
    id<MTLBuffer> outCosts = [_device newBufferWithLength:numImages*sizeof(float) options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> trainingDataBuffer = [_device newBufferWithLength:numImages*imageDataLength options:(MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared)];
    memcpy(trainingDataBuffer.contents, &trainData[0],numImages*imageDataLength);
    
    id<MTLBuffer> isCatDataBuffer = [_device newBufferWithLength:numImages options:MTLResourceStorageModeShared];
    memcpy(isCatDataBuffer.contents, &isCatData[0], numImages);
    
    _weights = [_device newBufferWithLength:(imageDataLength+1)*sizeof(float) options:MTLResourceStorageModeShared];
    memset(_weights.contents, 0, _weights.length);
    
    // Start Metal GPU capture
    MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
    captureDescriptor.captureObject = _device;
    captureDescriptor.destination = MTLCaptureDestinationDeveloperTools;
    
    NSError *captureError = nil;
    if ([captureManager startCaptureWithDescriptor:captureDescriptor error:&captureError]) {
        NSLog(@"Metal GPU capture started");
    } else {
        NSLog(@"Failed to start Metal GPU capture: %@", captureError.localizedDescription);
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
    [encoder dispatchThreads:MTLSizeMake(numImages, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
    
    // Stop Metal GPU capture
    [captureManager stopCapture];
    NSLog(@"Metal GPU capture stopped");
}
