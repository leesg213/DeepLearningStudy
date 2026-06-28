//
//  DeepHiddenLayerModel.cpp
//  PlanarDataClassificationWithDeepHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#include "DeepHiddenLayerModel.hpp"
#include "ShaderTypes.h"
#include <chrono>

static int capture_trace_iter = -1;

void DeepHiddenLayerModel::Init(std::vector<int> const& inLayers)
{
    Layers = inLayers;
    
    _device = MTLCreateSystemDefaultDevice();
    _cmdQueue = [_device newCommandQueue];
    
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
    
    NSError* error = nil;
    
    _pipelineDeepForwardRelu = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"deep_forward_relu"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    
    _pipelineDeepForwardSigmoid = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"deep_forward_sigmoid"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    
    _pipelineDeepComputeDAL = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"deep_compute_dAL"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    
    _pipelineDeepComputeGradRelu = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"deep_compute_grad_relu"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    
    _pipelineDeepComputeGradSigmoid = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"deep_compute_grad_sigmoid"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    
    _pipelineOptimize = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"optimize"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    
    _pipelineClearGrads = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"clear_grads"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    
    // Initialize random number generator with a random seed
    _rng = std::mt19937(std::random_device{}());
    _normalDist = std::normal_distribution<float>(0.0f, 1.0f);
    _rngInitialized = true;
    
    
}

float DeepHiddenLayerModel::ComputeCost(id<MTLBuffer> activation, size_t activationOffset, std::vector<uint8_t> const& label)
{
    float* activation_data = ((float*)activation.contents) + activationOffset;
    
    float cost = 0;
    for(int i = 0;i<label.size();++i)
    {
        float a = activation_data[i];
        cost += label[i] * log(a) + ( 1-label[i] ) * log(1-a);
    }
    
    cost = -cost/label.size();
    return cost;
}
void DeepHiddenLayerModel::PredictF(std::vector<float> const& test_x,
                                      int num_tests,
                                      std::vector<float>& out_results)
{
    out_results.clear();

    int num_total_activations = 0;
    int num_total_weights = 0;
    for(int i = 1;i<Layers.size();++i)
    {
        num_total_activations += Layers[i] * num_tests;
        num_total_weights += Layers[i] * ( Layers[i-1] + 1);
    }
    id<MTLBuffer> layerIDs_buffer = [_device newBufferWithLength:sizeof(int)*Layers.size() options:MTLResourceStorageModeShared];
    for(int i = 0;i<Layers.size();++i)
    {
        *(((int*)layerIDs_buffer.contents)+i) = i;
    }
    
    id<MTLBuffer> layers_buffer = [_device newBufferWithLength:Layers.size()*sizeof(int) options:MTLResourceStorageModeShared];
    memcpy(layers_buffer.contents, Layers.data(), layers_buffer.length);
    
    id<MTLBuffer> outActivations = [_device newBufferWithLength:num_total_activations*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> Zvalues = [_device newBufferWithLength:num_total_activations*sizeof(float) options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> trainingDataBuffer = [_device newBufferWithLength:test_x.size()*sizeof(float) options:(MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared)];
    memcpy(trainingDataBuffer.contents, test_x.data(), test_x.size()*sizeof(float));
    
    id<MTLBuffer> weights = [_device newBufferWithLength:num_total_weights*sizeof(float) options:MTLResourceStorageModeShared];
    memcpy(weights.contents, _weights.data(), weights.length);
    
    int last_activation_offset = 0;
    for(int i = 1;i<Layers.size()-1;++i)
    {
        last_activation_offset += num_tests * Layers[i];
    }
    
    id<MTLCommandBuffer> cmdBuffer = [_cmdQueue commandBuffer];
    cmdBuffer.label = @"Training Command Buffer";

    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    encoder.label = @"Training Compute Encoder";
    
    // Forward
    {
        int weight_offset = 0;
        int activation_offset = 0;
        
        for(int layer = 1;layer<Layers.size();++layer)
        {
            [encoder pushDebugGroup:[NSString stringWithFormat:@"[Forward Layer %d]", layer]];
            
            [encoder setComputePipelineState:(layer < Layers.size()-1) ? _pipelineDeepForwardRelu : _pipelineDeepForwardSigmoid];
            
            if(layer == 1)
            {
                [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
            }
            else
            {
                [encoder setBuffer:outActivations offset:(activation_offset - num_tests*Layers[layer-1])*sizeof(float) atIndex:0];
            }
            [encoder setBuffer:weights offset:weight_offset*sizeof(float) atIndex:1];
            [encoder setBuffer:outActivations offset:activation_offset*sizeof(float) atIndex:2];
            [encoder setBuffer:Zvalues offset:activation_offset*sizeof(float) atIndex:3];
            [encoder setBuffer:layerIDs_buffer offset:layer*sizeof(int) atIndex:4];
            [encoder setBuffer:layers_buffer offset:0 atIndex:5];
            
            [encoder dispatchThreads:MTLSizeMake(num_tests * Layers[layer], 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            
            weight_offset += Layers[layer] * (Layers[layer-1]+1);
            activation_offset += num_tests * Layers[layer];
            
            [encoder popDebugGroup];
        }
    }
    
    [encoder endEncoding];

    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];

    out_results.clear();
    float* activation_values = (float*)outActivations.contents;
    for(int i = 0;i<num_tests;++i)
    {
        float activation = activation_values[last_activation_offset + i];
        out_results.push_back(activation);
    }

}


void DeepHiddenLayerModel::TrainF(std::vector<float> const& train_x,
                                    std::vector<uint8_t> const& train_y,
                                    int num_trains,
                                    int numIterations,
                                    float learningRate)
{
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    id<MTLBuffer> uniforms = [_device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    Uniforms* uniforms_data = (Uniforms*)uniforms.contents;
    uniforms_data->numImages = num_trains;
    uniforms_data->numFeatures = 0;
    uniforms_data->normalizer_scaler = 1;
    uniforms_data->hidden_layer_size = 0;
    uniforms_data->learning_rate = learningRate;
    
    int num_total_activations = 0;
    int num_total_weights = 0;
    for(int i = 1;i<Layers.size();++i)
    {
        num_total_activations += Layers[i] * num_trains;
        num_total_weights += Layers[i] * ( Layers[i-1] + 1);
    }
    id<MTLBuffer> layerIDs_buffer = [_device newBufferWithLength:sizeof(int)*Layers.size() options:MTLResourceStorageModeShared];
    for(int i = 0;i<Layers.size();++i)
    {
        *(((int*)layerIDs_buffer.contents)+i) = i;
    }
    
    id<MTLBuffer> layers_buffer = [_device newBufferWithLength:Layers.size()*sizeof(int) options:MTLResourceStorageModeShared];
    memcpy(layers_buffer.contents, Layers.data(), layers_buffer.length);
    
    id<MTLBuffer> outCosts = [_device newBufferWithLength:num_trains*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outActivations = [_device newBufferWithLength:num_total_activations*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outGrads = [_device newBufferWithLength:num_total_weights*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> Zvalues = [_device newBufferWithLength:num_total_activations*sizeof(float) options:MTLResourceStorageModeShared];
    int num_total_dActivations = num_total_activations + Layers[0] * num_trains;
    id<MTLBuffer> dActivations = [_device newBufferWithLength:num_total_dActivations*sizeof(float) options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> trainingDataBuffer = [_device newBufferWithLength:train_x.size()*sizeof(float) options:(MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared)];
    memcpy(trainingDataBuffer.contents, train_x.data(), train_x.size()*sizeof(float));
    
    id<MTLBuffer> train_y_data_buffer = [_device newBufferWithLength:num_trains options:MTLResourceStorageModeShared];
    memcpy(train_y_data_buffer.contents, &train_y[0], num_trains);
    
    id<MTLBuffer> weights = [_device newBufferWithLength:num_total_weights*sizeof(float) options:MTLResourceStorageModeShared];
    std::vector<float> random_weights = randn(num_total_weights);
    memcpy(weights.contents, random_weights.data(), weights.length);
    float* weights_values = (float*)weights.contents;
    {
        int weight_offset = 0;
        for(int layer = 1;layer<Layers.size();++layer)
        {
            int weights_per_node = Layers[layer-1]+1;
            
            for(int j = 0;j<Layers[layer];++j)
            {
                for(int k = 0;k<weights_per_node-1;++k)
                {
                    weights_values[weight_offset+k+1] /= sqrt(Layers[layer-1]);
                }
                weights_values[weight_offset] = 0; // set b to 0
                weight_offset += weights_per_node;
            }
        }
    }
    
    int last_activation_offset = 0;
    for(int i = 1;i<Layers.size()-1;++i)
    {
        last_activation_offset += num_trains * Layers[i];
    }
    
    id<MTLCommandBuffer> cmdBuffer = nil;
    
    for(int iter = 0;iter < numIterations; ++iter)
    {
        if(capture_trace_iter == -1)
        {
            if(iter%100 == 0)
            {
                if(cmdBuffer)
                {
                    [cmdBuffer commit];
                    [cmdBuffer waitUntilCompleted];
                    
                    float cost = ComputeCost(outActivations, last_activation_offset, train_y);
                    NSLog(@"[%d] Cost : %f",iter, cost);
                }
                
                cmdBuffer = [_cmdQueue commandBuffer];
                cmdBuffer.label = @"Training Command Buffer";
            }
        }
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
        
        if(capture_trace_iter>=0)
        {
            cmdBuffer = [_cmdQueue commandBuffer];
            cmdBuffer.label = @"Training Command Buffer";
        }

        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        encoder.label = @"Training Compute Encoder";
        
        // Forward
        {
            int weight_offset = 0;
            int activation_offset = 0;
            
            for(int layer = 1;layer<Layers.size();++layer)
            {
                [encoder pushDebugGroup:[NSString stringWithFormat:@"[Forward Layer %d]", layer]];
                
                [encoder setComputePipelineState:(layer < Layers.size()-1) ? _pipelineDeepForwardRelu : _pipelineDeepForwardSigmoid];
                
                if(layer == 1)
                {
                    [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
                }
                else
                {
                    [encoder setBuffer:outActivations offset:(activation_offset - num_trains*Layers[layer-1])*sizeof(float) atIndex:0];
                }
                [encoder setBuffer:weights offset:weight_offset*sizeof(float) atIndex:1];
                [encoder setBuffer:outActivations offset:activation_offset*sizeof(float) atIndex:2];
                [encoder setBuffer:Zvalues offset:activation_offset*sizeof(float) atIndex:3];
                [encoder setBuffer:layerIDs_buffer offset:layer*sizeof(int) atIndex:4];
                [encoder setBuffer:layers_buffer offset:0 atIndex:5];
                
                [encoder dispatchThreads:MTLSizeMake(num_trains * Layers[layer], 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                
                weight_offset += Layers[layer] * (Layers[layer-1]+1);
                activation_offset += num_trains * Layers[layer];
                
                [encoder popDebugGroup];
            }
        }
        
        // Backward
        [encoder setComputePipelineState:_pipelineClearGrads];
        [encoder setBuffer:outGrads offset:0 atIndex: 0];
        [encoder dispatchThreads:MTLSizeMake(num_total_weights, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder setBuffer:dActivations offset:0 atIndex: 0];
        [encoder dispatchThreads:MTLSizeMake(num_total_dActivations, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [encoder setComputePipelineState:_pipelineDeepComputeDAL];
        [encoder setBuffer:outActivations offset:sizeof(float)*last_activation_offset atIndex:0];
        [encoder setBuffer:train_y_data_buffer offset:0 atIndex:1];
        [encoder setBuffer:dActivations offset:sizeof(float)*(num_total_dActivations-num_trains) atIndex:2];

        [encoder dispatchThreads:MTLSizeMake(num_trains, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        
        {
            int weight_offset = num_total_weights - Layers[Layers.size()-1] * (Layers[Layers.size()-2]+1);
            int activation_offset = num_total_activations - Layers[Layers.size()-1] * num_trains;
            int dActivation_offset = num_total_dActivations - Layers[Layers.size()-1] * num_trains;
            for(int layer = Layers.size()-1;layer>=1;--layer)
            {
                [encoder pushDebugGroup:[NSString stringWithFormat:@"[Backward Layer %d]", layer]];
                
                [encoder setComputePipelineState:(layer < Layers.size()-1) ? _pipelineDeepComputeGradRelu : _pipelineDeepComputeGradSigmoid];
                
                [encoder setBuffer:outActivations offset:sizeof(float)*(activation_offset) atIndex:0];
                
                if(layer == 1)
                {
                    [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 1];
                }
                else
                {
                    [encoder setBuffer:outActivations offset:sizeof(float)*(activation_offset-num_trains * Layers[layer-1]) atIndex:1];
                }
                
                [encoder setBuffer:dActivations offset:sizeof(float)*dActivation_offset atIndex:2];
                [encoder setBuffer:dActivations offset:sizeof(float)*(dActivation_offset - num_trains * Layers[layer-1]) atIndex:3];
                [encoder setBuffer:Zvalues offset:activation_offset*sizeof(float) atIndex:4];
                [encoder setBuffer:weights offset:sizeof(float)*weight_offset atIndex:5];
                [encoder setBuffer:layerIDs_buffer offset:layer*sizeof(int) atIndex:6];
                [encoder setBuffer:layers_buffer offset:0 atIndex:7];
                [encoder setBuffer:outGrads offset:weight_offset*sizeof(float) atIndex:8];
                
                [encoder dispatchThreads:MTLSizeMake(num_trains* Layers[layer], 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                
                if(layer>1)
                {
                    weight_offset -= Layers[layer-1] * (Layers[layer-2]+1);
                    activation_offset -= num_trains * Layers[layer-1];
                    dActivation_offset -= num_trains * Layers[layer-1];
                }
                [encoder popDebugGroup];
            }
        }
        
        [encoder setComputePipelineState:_pipelineOptimize];
        [encoder setBuffer:weights offset:0 atIndex: 0];
        [encoder setBuffer:outGrads offset:0 atIndex: 1];
        [encoder setBuffer:uniforms offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(num_total_weights, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        
        
        [encoder endEncoding];
  
        if(capture_trace_iter>=0)
        {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            
            float cost = ComputeCost(outActivations, last_activation_offset, train_y);
            NSLog(@"[%d] Cost : %f",iter, cost);
        }
        
        if(captureManager)
        {
            // Stop Metal GPU capture
            [captureManager stopCapture];
            NSLog(@"Metal GPU capture stopped");
        }
    }
    
    if(capture_trace_iter == -1)
    {
        if(cmdBuffer)
        {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }
    }
    
    _weights.clear();
    {
        float* weight_values = (float*)weights.contents;
        for(int i = 0;i<num_total_weights;++i)
        {
            _weights.push_back(weight_values[i]);
        }
    }

    // End timing and calculate duration
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Log timing results
    double seconds = duration.count() / 1000.0;
    NSLog(@"====================================");
    NSLog(@"Train complete");
    NSLog(@"Total training time: %.3f seconds (%.2f ms)", seconds, (double)duration.count());
    NSLog(@"Time per iteration: %.3f ms", (double)duration.count() / numIterations);
    NSLog(@"====================================");
}
#pragma mark - Random Number Generation (np.random.randn port)

void DeepHiddenLayerModel::seed(unsigned int seedValue)
{
    _rng.seed(seedValue);
    _normalDist.reset();
    _rngInitialized = true;
}

float DeepHiddenLayerModel::randn()
{
    if (!_rngInitialized) {
        // Initialize with random seed if not already initialized
        _rng = std::mt19937(std::random_device{}());
        _normalDist = std::normal_distribution<float>(0.0f, 1.0f);
        _rngInitialized = true;
    }
    return _normalDist(_rng);
}

std::vector<float> DeepHiddenLayerModel::randn(size_t size)
{
    if (!_rngInitialized) {
        // Initialize with random seed if not already initialized
        _rng = std::mt19937(std::random_device{}());
        _normalDist = std::normal_distribution<float>(0.0f, 1.0f);
        _rngInitialized = true;
    }
    
    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = _normalDist(_rng);
    }
    return result;
}

