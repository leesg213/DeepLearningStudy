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


kernel void computeCostF_1HiddenLayer(device float* in_train_data_x [[buffer(0)]],
                        device uint8_t* in_train_data_y [[buffer(1)]],
                        device float* weights [[buffer(2)]],
                        constant Uniforms& uniforms [[buffer(3)]],
                        device float* outCosts [[buffer(4)]],
                        device float* outActivations [[buffer(5)]],
                        uint thread_id [[thread_position_in_grid]])
{
    device float* train_data_x = &in_train_data_x[uniforms.numFeatures*thread_id];
    uint8_t train_data_y = in_train_data_y[thread_id];
    
    const int num_weights_per_hidden_node = uniforms.numFeatures+1;
    const int num_activations_per_train = uniforms.numHiddenLayers+1;
    
    for(uint32_t h = 0;h<uniforms.numHiddenLayers;++h)
    {
        float Z1 = 0;
        for(uint32_t i = 0;i<uniforms.numFeatures;++i)
        {
            float x = train_data_x[i] * uniforms.normalizer_scaler;
            float w = weights[h*num_weights_per_hidden_node+i+1];
            float m = x * w;
            Z1 += m;
        }
        Z1 += weights[h*num_weights_per_hidden_node];
        
        float A1 = metal::tanh(Z1);
        outActivations[thread_id*num_activations_per_train+h] = A1;
    }
    
    // Calc activation
    const int last_weight_start_pos = num_weights_per_hidden_node*uniforms.numHiddenLayers;
    float Z2 = 0;
    for(uint32_t i = 0;i<uniforms.numHiddenLayers;++i)
    {
        float x = outActivations[thread_id*num_activations_per_train+i];
        float w = weights[last_weight_start_pos+i+1];
        float m = x * w;
        Z2 += m;
    }
    Z2 += weights[last_weight_start_pos];
    float A2 = sigmoid(Z2);
    
    outActivations[thread_id*num_activations_per_train+uniforms.numHiddenLayers] = A2;
    
    float cost = train_data_y*metal::log(A2)+(1-train_data_y)*metal::log(1-A2);
    // Calc cost
    outCosts[thread_id] = cost;
}

kernel void computeGradsF_1HiddenLayer(device float* train_data_x [[buffer(0)]],
                        device uint8_t* train_data_y [[buffer(1)]],
                        device float* allActivations [[buffer(2)]],
                        device float* outGrads [[buffer(3)]],
                       device float* weights [[buffer(4)]],
                        constant Uniforms& uniforms [[buffer(5)]],
                        uint thread_id [[thread_position_in_grid]])
{
    const int num_total_weights = (uniforms.numFeatures+1) * uniforms.numHiddenLayers + (uniforms.numHiddenLayers + 1);
    const int num_weights_per_hidden_node = uniforms.numFeatures+1;
    const int num_activations_per_train = uniforms.numHiddenLayers+1;
    const int last_weight_start_pos = num_weights_per_hidden_node*uniforms.numHiddenLayers;
    
    if(thread_id >= last_weight_start_pos)
    {
        int local_thread_id = thread_id - last_weight_start_pos;
        
        float sum = 0;
        for(int i = 0;i<uniforms.numImages;++i)
        {
            float a2 = allActivations[i*num_activations_per_train+num_activations_per_train-1];
            float a1 = local_thread_id == 0 ? 1 : allActivations[i*num_activations_per_train+local_thread_id-1];
            float y = train_data_y[i];
            float dZ2 = (a2 - y);
            sum += a1 * dZ2;
        }
        
        outGrads[thread_id] = sum / uniforms.numImages;
        return;
    }
    
    float sum = 0;
    for(int i = 0;i<uniforms.numImages;++i)
    {
        int hidden_node_id = thread_id/num_weights_per_hidden_node;
        int local_thread_id = thread_id%num_weights_per_hidden_node;
        
        float a2 = allActivations[i*num_activations_per_train+num_activations_per_train-1];
        float y = train_data_y[i];
        float dZ2 = (a2 - y);
        float W2 = weights[last_weight_start_pos+hidden_node_id+1];
        float a1 = allActivations[i*num_activations_per_train+hidden_node_id];
        
        float x = local_thread_id == 0 ? 1 : train_data_x[i*uniforms.numFeatures+local_thread_id-1];
        float dZ1 = W2 * dZ2 * ( 1 - metal::pow(a1, 2.0f));
        sum += dZ1 * x;
    }
    
    outGrads[thread_id] = sum / uniforms.numImages;
}
