#include "../include/optimizer_sgd.hpp"
#include "../include/utils.hpp"

Optimizer_SGD::Optimizer_SGD(float learning_rate, float decay, float momentum) 
    : learning_rate(learning_rate), decay(decay), current_learning_rate(learning_rate), 
      momentum(momentum), iterations(0) {}

void Optimizer_SGD::pre_update_params() {
    // if there is any decay, then the current learning rate gets updated, based on the decay times the iterations
    if (decay > 0)
    {
        current_learning_rate = learning_rate / (1.0f + decay * iterations);
    }
    
}

void Optimizer_SGD::update_params(LayerDense &layer) {
    if (momentum != 0)
    {
        if (layer.weight_momentums.cols() == 0 || layer.weight_momentums.rows() == 0)
        {
            layer.weight_momentums = create_matrix(layer.weights.rows(), layer.weights.cols(), 0.0);
            layer.biases_momentum = create_matrix(1, layer.biases.cols(), 0.0);
        }

        FlatMatrix weight_updates = momentum * layer.weight_momentums - current_learning_rate * layer.dweights;
        layer.weight_momentums = weight_updates;
        layer.weights = layer.weights + weight_updates;
        
        FlatMatrix bias_updates = momentum * layer.biases_momentum - current_learning_rate * layer.dbiases;
        layer.biases_momentum = bias_updates;
        layer.biases = layer.biases + bias_updates;
    }
    else
    {
        // Update weights: weights = weights - learning_rate * dweights
        layer.weights = layer.weights - current_learning_rate * layer.dweights;
        
        // Update biases: biases = biases - learning_rate * dbiases
        layer.biases = layer.biases - current_learning_rate * layer.dbiases;
    }
}

void Optimizer_SGD::post_update_params() {
    iterations += 1;
}