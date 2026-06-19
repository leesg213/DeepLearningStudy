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
    return 1.0f / (1 + metal::exp(-inValue));
}
kernel void computeCost(device uint8_t* training_data [[buffer(0)]],
                        device uint8_t* isCat_data [[buffer(1)]],
                        device float* weights [[buffer(2)]],
                        constant Uniforms& uniforms [[buffer(3)]],
                        device float* outCosts [[buffer(4)]],
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
    
    // Calc cost
    outCosts[thread_id] = isCat*metal::log(A)+(1-isCat)*metal::log(1-A);
}
kernel void network(device uint8_t& training_data [[buffer(0)]],
                    device float& weights [[buffer(1)]])
{
    
}
