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
    const int num_activations_per_train = uniforms.hidden_layer_size+1;
    
    for(uint32_t h = 0;h<uniforms.hidden_layer_size;++h)
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
    const int last_weight_start_pos = num_weights_per_hidden_node*uniforms.hidden_layer_size;
    float Z2 = 0;
    for(uint32_t i = 0;i<uniforms.hidden_layer_size;++i)
    {
        float x = outActivations[thread_id*num_activations_per_train+i];
        float w = weights[last_weight_start_pos+i+1];
        float m = x * w;
        Z2 += m;
    }
    Z2 += weights[last_weight_start_pos];
    float A2 = sigmoid(Z2);
    
    outActivations[thread_id*num_activations_per_train+uniforms.hidden_layer_size] = A2;
    
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
    const int num_total_weights = (uniforms.numFeatures+1) * uniforms.hidden_layer_size + (uniforms.hidden_layer_size + 1);
    const int num_weights_per_hidden_node = uniforms.numFeatures+1;
    const int num_activations_per_train = uniforms.hidden_layer_size+1;
    const int last_weight_start_pos = num_weights_per_hidden_node*uniforms.hidden_layer_size;
    
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


kernel void computeGradsF_1HiddenLayer_V2(device float* train_data_x [[buffer(0)]],
                        device uint8_t* train_data_y [[buffer(1)]],
                        device float* allActivations [[buffer(2)]],
                        device atomic_float* outGrads [[buffer(3)]],
                       device float* weights [[buffer(4)]],
                        constant Uniforms& uniforms [[buffer(5)]],
                        uint thread_id [[thread_position_in_grid]])
{
    const int num_total_weights = (uniforms.numFeatures+1) * uniforms.hidden_layer_size + (uniforms.hidden_layer_size + 1);
    const int num_weights_per_hidden_node = uniforms.numFeatures+1;
    const int num_activations_per_train = uniforms.hidden_layer_size+1;
    const int last_weight_start_pos = num_weights_per_hidden_node*uniforms.hidden_layer_size;
    
    float a2 = allActivations[thread_id*num_activations_per_train+num_activations_per_train-1];
    float y = train_data_y[thread_id];
    float dZ2 = (a2 - y);

    for(int weight_id = 0;weight_id<uniforms.hidden_layer_size+1;++weight_id)
    {
        float a1 = weight_id == 0 ? 1 : allActivations[thread_id*num_activations_per_train+weight_id-1];
        float grad = a1 * dZ2;
        
        atomic_fetch_add_explicit(outGrads+last_weight_start_pos+weight_id, grad, memory_order_relaxed);
    }
    
    for(int hidden_node_id = 0;hidden_node_id<uniforms.hidden_layer_size;++hidden_node_id)
    {
        float W2 = weights[last_weight_start_pos+hidden_node_id+1];
        float a1 = allActivations[thread_id*num_activations_per_train+hidden_node_id];
        float dZ1 = W2 * dZ2 * ( 1 - metal::pow(a1, 2.0f));
        for(int weight_id = 0;weight_id<uniforms.numFeatures+1;++weight_id)
        {
            float x = weight_id == 0 ? 1 : train_data_x[thread_id*uniforms.numFeatures+weight_id-1];
            float grad = dZ1 * x;
            
            atomic_fetch_add_explicit(outGrads+hidden_node_id*num_weights_per_hidden_node+weight_id, grad, memory_order_relaxed);
        }
    }
}

kernel void clear_grads(device float* outGrads [[buffer(0)]],
                        uint thread_id [[thread_position_in_grid]])
{
    outGrads[thread_id] = 0;
}

kernel void optimize(device float* weights [[buffer(0)]],
                     device float* outGrads [[buffer(1)]],
                     constant Uniforms& uniforms [[buffer(2)]],
                     uint thread_id [[thread_position_in_grid]])
{
    weights[thread_id] = weights[thread_id] - uniforms.learning_rate * outGrads[thread_id] / uniforms.numImages;
}

kernel void deep_forward(device float* in_A_Prev [[buffer(0)]],
                        device float* weights [[buffer(1)]],
                        device float* outActivations [[buffer(2)]],
                         device float* outZvalues [[buffer(3)]],
                        device int& layerID [[buffer(4)]],
                        device int* layers [[buffer(5)]],
                         device int& activation_type [[buffer(6)]], // activation_type 0 : relu, 1 : sigmoid
                        uint thread_id [[thread_position_in_grid]])
{
    uint n_current = layers[layerID];
    uint dataset_id = (uint)(thread_id/n_current);
    uint node_id = thread_id%n_current;
    uint n_prev = layers[layerID - 1];
    
    uint num_weights_per_node = (n_prev+1);
    
    float z = 0;
    for(uint j = 0;j<n_prev+1;++j)
    {
        float a_prev = j == 0 ? 1 : in_A_Prev[n_prev*dataset_id+j-1];
        z += weights[num_weights_per_node*node_id+j] * a_prev;
    }
    
    float a = activation_type == 0 ? max(0.0f, z) : sigmoid(z);
    
    outZvalues[n_current*dataset_id+node_id] = z;
    outActivations[n_current*dataset_id+node_id] = a;
}

kernel void deep_compute_dAL(device float* activations [[buffer(0)]],
                        device uint8_t* labels [[buffer(1)]],
                         device float* outdActivations [[buffer(2)]],
                        uint thread_id [[thread_position_in_grid]])
{
    uint dataset_id = thread_id;
    uint8_t Y = labels[dataset_id];
    float a = activations[dataset_id];
    
    outdActivations[dataset_id] = -((Y / a) - (1-Y)/(1-a));
}

float relu_backward(float Z, float dA)
{
    return Z<=0 ? 0 : dA;
}

float sigmoid_backward(float Z, float dA)
{
    float s = sigmoid(Z);
    return dA * s * (1-s);
}


kernel void deep_compute_grad(device float* activations [[buffer(0)]],
                                      device float* activations_prev [[buffer(1)]],
                         device float* dActivations [[buffer(2)]],
                           device atomic_float* dActivationsPrevLayer [[buffer(3)]],
                                      device float* Zvalues [[buffer(4)]],
                                      device float* weights [[buffer(5)]],
                                      device int& layerID [[buffer(6)]],
                                      device int* layers [[buffer(7)]],
                                      device atomic_float* outGrads [[buffer(8)]],
                                   device int& activation_type [[buffer(9)]], // activation_type 0 : relu, 1 : sigmoid
                        uint thread_id [[thread_position_in_grid]])
{
    uint n_current = layers[layerID];
    uint dataset_id = (uint)(thread_id/n_current);
    uint node_id = thread_id%n_current;
    uint n_prev = layers[layerID - 1];
    uint num_weights_per_node = (n_prev+1);
    
    uint grad_index = node_id*num_weights_per_node;
    
    float Z = Zvalues[dataset_id*n_current+node_id];
    float dA = dActivations[dataset_id*n_current+node_id];
    float dZ = activation_type == 0 ? relu_backward(Z, dA) : sigmoid_backward(Z, dA);
    
    {
        float grad = dZ;
        atomic_fetch_add_explicit(outGrads+grad_index, grad, memory_order_relaxed);
        ++grad_index;
    }
    
    for(int weight_id = 0;weight_id<n_prev;++weight_id)
    {
        float a_prev = activations_prev[dataset_id*n_prev+weight_id];
        float grad = a_prev * dZ;
        
        atomic_fetch_add_explicit(outGrads+grad_index, grad, memory_order_relaxed);
        
        if(layerID > 1)
        {
            float dA_Prev = dZ * weights[num_weights_per_node*node_id+weight_id+1];
            atomic_fetch_add_explicit(dActivationsPrevLayer+dataset_id*n_prev+weight_id, dA_Prev, memory_order_relaxed);
        }
        
        ++grad_index;
    }
}
