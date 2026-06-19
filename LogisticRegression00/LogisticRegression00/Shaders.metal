//
//  Shaders.metal
//  LogisticRegression00
//
//  Created by Sugu Lee on 6/19/26.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>
#include "ShaderTypes.h"
#include <metal_math>

float sigmod(float inValue)
{
    float epsilon = 0.000001f;
    return metal::clamp((1.0f / (1 + metal::exp(-inValue))),epsilon,1-epsilon);
}
kernel void computeCost(device uint8_t* training_data [[buffer(0)]],
                        device uint8_t* isCat_data [[buffer(1)]],
                        device float* weights [[buffer(2)]],
                        constant Uniforms& uniforms [[buffer(3)]],
                        device float* outCosts [[buffer(4)]],
                        device float* outActivations [[buffer(5)]],
                        uint thread_id [[thread_position_in_grid]])
{
    device uint8_t* current_training_data = &training_data[uniforms.imageDataLength*thread_id];
    uint8_t isCat = isCat_data[thread_id];
    
    // Calc activation
    float A = 0;
    for(uint32_t i = 0;i<uniforms.imageDataLength;++i)
    {
        A += current_training_data[i]/255.0f * weights[i+1];
    }
    A += weights[0];
    A = sigmod(A);
    
    outActivations[thread_id] = A;
    
    float cost = isCat*metal::log(A)+(1-isCat)*metal::log(1-A);
    // Calc cost
    outCosts[thread_id] = cost;
}

kernel void computeGrads(device uint8_t* training_data [[buffer(0)]],
                        device uint8_t* isCat_data [[buffer(1)]],
                        device float* allActivations [[buffer(2)]],
                        device float* outGrads [[buffer(3)]],
                        constant Uniforms& uniforms [[buffer(4)]],
                        uint thread_id [[thread_position_in_grid]])
{
    float dotProduct = 0;
    for(int i = 0;i<uniforms.numImages;++i)
    {
        float training_value = thread_id == 0 ? 1 : (training_data[i*uniforms.imageDataLength+(thread_id-1)]/255.0f);
        dotProduct += training_value * (allActivations[i] - isCat_data[i]);
    }
    dotProduct /= uniforms.numImages;
    outGrads[thread_id] = dotProduct;
}

kernel void network(device uint8_t& training_data [[buffer(0)]],
                    device float& weights [[buffer(1)]])
{
    
}
