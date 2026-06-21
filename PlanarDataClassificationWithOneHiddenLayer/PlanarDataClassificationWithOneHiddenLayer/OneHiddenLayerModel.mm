//
//  OneHiddenLayerModel.cpp
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#include "OneHiddenLayerModel.hpp"
#include "ShaderTypes.h"
#include <chrono>

static int capture_trace_iter = -1;

void OneHiddenLayerModel::Init()
{
    _device = MTLCreateSystemDefaultDevice();
    _cmdQueue = [_device newCommandQueue];
    
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
    
    NSError* error = nil;
    
    _pipelineComputeCostF_1HiddenLayer = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"computeCostF_1HiddenLayer"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    
    _pipelineComputeGradsF_1HiddenLayer = [_device newComputePipelineStateWithFunction:[defaultLibrary newFunctionWithName:@"computeGradsF_1HiddenLayer_V2"] error:&error];
    if(error != nil)
    {
        NSLog(@"Failed to load pipeline. %@", error.localizedDescription);
    }
    
    // Initialize random number generator with a random seed
    _rng = std::mt19937(std::random_device{}());
    _normalDist = std::normal_distribution<float>(0.0f, 1.0f);
    _rngInitialized = true;
}

float OneHiddenLayerModel::CalcCost(id<MTLBuffer> allCosts, size_t num_trains)
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
void OneHiddenLayerModel::PredictF(std::vector<float> const& test_x,
                                   int num_hidden_layers,
                                      int num_tests,
                                      int num_features,
                                      std::vector<float>& out_results)
{
    
    out_results.clear();
    
    id<MTLBuffer> uniforms = [_device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    Uniforms* uniforms_data = (Uniforms*)uniforms.contents;
    uniforms_data->numImages = num_tests;
    uniforms_data->numFeatures = num_features;
    uniforms_data->normalizer_scaler = 1;
    uniforms_data->numHiddenLayers = num_hidden_layers;
    
    int const num_total_weights = ((num_features+1)*num_hidden_layers+(num_hidden_layers+1));

    id<MTLBuffer> outCosts = [_device newBufferWithLength:num_tests*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outActivations = [_device newBufferWithLength:(num_tests*(num_hidden_layers+1))*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outGrads = [_device newBufferWithLength:num_total_weights*sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLBuffer> trainingDataBuffer = [_device newBufferWithLength:test_x.size()*sizeof(float) options:(MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared)];
    memcpy(trainingDataBuffer.contents, test_x.data(), test_x.size()*sizeof(float));
    
    id<MTLBuffer> train_y_data_buffer = [_device newBufferWithLength:num_tests options:MTLResourceStorageModeShared];
    memset(train_y_data_buffer.contents, 0, num_tests);
    
    id<MTLBuffer> weights = [_device newBufferWithLength:num_total_weights*sizeof(float) options:MTLResourceStorageModeShared];
    memcpy(weights.contents, _weights.data(), weights.length);
    
    // Start Metal GPU capture
    MTLCaptureManager *captureManager = nil;
    if(false)
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
    
    [encoder setComputePipelineState:_pipelineComputeCostF_1HiddenLayer];
    [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
    [encoder setBuffer:train_y_data_buffer offset:0 atIndex: 1];
    [encoder setBuffer:weights offset:0 atIndex:2];
    [encoder setBuffer:uniforms offset:0 atIndex:3];
    [encoder setBuffer:outCosts offset:0 atIndex:4];
    [encoder setBuffer:outActivations offset:0 atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(num_tests, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];

    if(captureManager)
    {
        // Stop Metal GPU capture
        [captureManager stopCapture];
        NSLog(@"Metal GPU capture stopped");
    }

    out_results.clear();
    float* activation_values = (float*)outActivations.contents;
    for(int i = 0;i<num_tests;++i)
    {
        float activation = activation_values[i*(num_hidden_layers+1)+num_hidden_layers];
        out_results.push_back(activation);
    }
}


void OneHiddenLayerModel::TrainF(std::vector<float> const& train_x,
                                    std::vector<uint8_t> const& train_y,
                                 int num_hidden_layers,
                                    int num_trains,
                                    int num_features,
                                    int numIterations,
                                    float learningRate)
{
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    id<MTLBuffer> uniforms = [_device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    Uniforms* uniforms_data = (Uniforms*)uniforms.contents;
    uniforms_data->numImages = num_trains;
    uniforms_data->numFeatures = num_features;
    uniforms_data->normalizer_scaler = 1;
    uniforms_data->numHiddenLayers = num_hidden_layers;
    
    int const num_total_weights = ((num_features+1)*num_hidden_layers+(num_hidden_layers+1));

    id<MTLBuffer> outCosts = [_device newBufferWithLength:num_trains*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outActivations = [_device newBufferWithLength:(num_trains*(num_hidden_layers+1))*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outGrads = [_device newBufferWithLength:num_total_weights*sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLBuffer> trainingDataBuffer = [_device newBufferWithLength:train_x.size()*sizeof(float) options:(MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared)];
    memcpy(trainingDataBuffer.contents, train_x.data(), train_x.size()*sizeof(float));
    
    id<MTLBuffer> train_y_data_buffer = [_device newBufferWithLength:num_trains options:MTLResourceStorageModeShared];
    memcpy(train_y_data_buffer.contents, &train_y[0], num_trains);
    
    id<MTLBuffer> weights = [_device newBufferWithLength:num_total_weights*sizeof(float) options:MTLResourceStorageModeShared];
    std::vector<float> random_weights = randn(num_total_weights);
    for(int i = 0;i<random_weights.size();++i)
    {
        random_weights[i] *= 0.01f;
    }
    memcpy(weights.contents, random_weights.data(), weights.length);
    float* weights_values = (float*)weights.contents;
    for(int i = 0;i<num_hidden_layers;++i)
    {
        weights_values[i*(num_features+1)] = 0;
    }
    weights_values[(num_features+1)*num_hidden_layers] = 0;
    
    for(int iter = 0;iter < numIterations; ++iter)
    {
        NSLog(@"Iteration : %d", iter);
        
        memset(outGrads.contents, 0, outGrads.length); // reset grads
        
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
        
        [encoder setComputePipelineState:_pipelineComputeCostF_1HiddenLayer];
        [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
        [encoder setBuffer:train_y_data_buffer offset:0 atIndex: 1];
        [encoder setBuffer:weights offset:0 atIndex:2];
        [encoder setBuffer:uniforms offset:0 atIndex:3];
        [encoder setBuffer:outCosts offset:0 atIndex:4];
        [encoder setBuffer:outActivations offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(num_trains, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        
        [encoder setComputePipelineState:_pipelineComputeGradsF_1HiddenLayer];
        [encoder setBuffer:trainingDataBuffer offset:0 atIndex: 0];
        [encoder setBuffer:train_y_data_buffer offset:0 atIndex: 1];
        [encoder setBuffer:outActivations offset:0 atIndex: 2];
        [encoder setBuffer:outGrads offset:0 atIndex: 3];
        [encoder setBuffer:weights offset:0 atIndex:4];
        [encoder setBuffer:uniforms offset:0 atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(num_trains, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        float cost = CalcCost(outCosts, num_trains);
        NSLog(@"Cost : %f", cost);

        char weight_log[1000];
        weight_log[0] = 0;
        
        float* gradValues = (float*)outGrads.contents;
        // Update weights
        float* weight_values = (float*)weights.contents;
        for(int i = 0;i<num_total_weights;++i)
        {
            weight_values[i] = weight_values[i] - learningRate * gradValues[i] / num_trains;
           // strcat(weight_log, [NSString stringWithFormat:@"%f,", weight_values[i]].UTF8String);
        }
        //NSLog(@"Weights : %s", weight_log);

        if(captureManager)
        {
            // Stop Metal GPU capture
            [captureManager stopCapture];
            NSLog(@"Metal GPU capture stopped");
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

void OneHiddenLayerModel::seed(unsigned int seedValue)
{
    _rng.seed(seedValue);
    _normalDist.reset();
    _rngInitialized = true;
}

float OneHiddenLayerModel::randn()
{
    if (!_rngInitialized) {
        // Initialize with random seed if not already initialized
        _rng = std::mt19937(std::random_device{}());
        _normalDist = std::normal_distribution<float>(0.0f, 1.0f);
        _rngInitialized = true;
    }
    return _normalDist(_rng);
}

std::vector<float> OneHiddenLayerModel::randn(size_t size)
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

