//
//  Shaders.metal
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"
#include <metal_math>

float sigmoid(float inValue)
{
    float epsilon = 0.000001f;
    return metal::clamp((1.0f / (1 + metal::exp(-inValue))),epsilon,1-epsilon);
}
kernel void computeCost(device uint8_t* train_data_x [[buffer(0)]],
                        device uint8_t* train_data_y [[buffer(1)]],
                        device float* weights [[buffer(2)]],
                        constant Uniforms& uniforms [[buffer(3)]],
                        device float* outCosts [[buffer(4)]],
                        device float* outActivations [[buffer(5)]],
                        uint thread_id [[thread_position_in_grid]])
{
    device uint8_t* current_train_data_x = &train_data_x[uniforms.numFeatures*thread_id];
    uint8_t current_y = train_data_y[thread_id];
    
    // Calc activation
    float A = 0;
    for(uint32_t i = 0;i<uniforms.numFeatures;++i)
    {
        float x = current_train_data_x[i] * uniforms.normalizer_scaler;
        //float x = X[thread_id][i];;
        float w = weights[i+1];
        float m = x * w;
        A += m;
    }
    float b = weights[0];
    A += b;
    A = sigmoid(A);
    
    outActivations[thread_id] = A;
    
    float cost = current_y*metal::log(A)+(1-current_y)*metal::log(1-A);
    // Calc cost
    outCosts[thread_id] = cost;
}


kernel void computeGrads(device uint8_t* train_data_x [[buffer(0)]],
                        device uint8_t* train_data_y [[buffer(1)]],
                        device float* allActivations [[buffer(2)]],
                        device float* outGrads [[buffer(3)]],
                        constant Uniforms& uniforms [[buffer(4)]],
                        uint thread_id [[thread_position_in_grid]])
{

    float dotProduct = 0;
    for(int i = 0;i<uniforms.numImages;++i)
    {
        float training_value = thread_id == 0 ? 1 : (train_data_x[i*uniforms.numFeatures+(thread_id-1)] * uniforms.normalizer_scaler);
        
        dotProduct += training_value * (allActivations[i] - train_data_y[i]);
    }
    dotProduct /= uniforms.numImages;
    outGrads[thread_id] = dotProduct;
}

kernel void computeCostF(device float* train_data_x [[buffer(0)]],
                        device uint8_t* train_data_y [[buffer(1)]],
                        device float* weights [[buffer(2)]],
                        constant Uniforms& uniforms [[buffer(3)]],
                        device float* outCosts [[buffer(4)]],
                        device float* outActivations [[buffer(5)]],
                        uint thread_id [[thread_position_in_grid]])
{
    device float* current_train_data_x = &train_data_x[uniforms.numFeatures*thread_id];
    uint8_t current_y = train_data_y[thread_id];
    
    // Calc activation
    float A = 0;
    for(uint32_t i = 0;i<uniforms.numFeatures;++i)
    {
        float x = current_train_data_x[i] * uniforms.normalizer_scaler;
        //float x = X[thread_id][i];;
        float w = weights[i+1];
        float m = x * w;
        A += m;
    }
    float b = weights[0];
    A += b;
    A = sigmoid(A);
    
    outActivations[thread_id] = A;
    
    float cost = current_y*metal::log(A)+(1-current_y)*metal::log(1-A);
    // Calc cost
    outCosts[thread_id] = cost;
}


kernel void computeGradsF(device float* train_data_x [[buffer(0)]],
                        device uint8_t* train_data_y [[buffer(1)]],
                        device float* allActivations [[buffer(2)]],
                        device float* outGrads [[buffer(3)]],
                        constant Uniforms& uniforms [[buffer(4)]],
                        uint thread_id [[thread_position_in_grid]])
{

    float dotProduct = 0;
    for(int i = 0;i<uniforms.numImages;++i)
    {
        float training_value = thread_id == 0 ? 1 : (train_data_x[i*uniforms.numFeatures+(thread_id-1)] * uniforms.normalizer_scaler);
        dotProduct += training_value * (allActivations[i] - train_data_y[i]);
    }
    dotProduct /= uniforms.numImages;
    outGrads[thread_id] = dotProduct;
}
