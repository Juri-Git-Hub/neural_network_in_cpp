#include "../include/optimizer_adagrad.hpp"
#include "../include/utils.hpp"

Optimizer_Adagrad::Optimizer_Adagrad(float learning_rate, float decay, float eps) 
    : learning_rate(learning_rate), decay(decay), current_learning_rate(learning_rate), 
      eps(eps), iterations(0) {}

void Optimizer_Adagrad::pre_update_params() {
    // if there is any decay, then the current learning rate gets updated, based on the decay times the iterations
    if (decay > 0)
    {
        current_learning_rate = learning_rate / (1.0f + decay * iterations);
    }
    
}

void Optimizer_Adagrad::update_params(LayerDense &layer) {

    if (layer.weight_cache.cols() == 0 || layer.weight_cache.rows() == 0)
    {
        layer.weight_cache = create_matrix(layer.weights.rows(), layer.weights.cols(), 0.0);
        layer.bias_cache = create_matrix(1, layer.biases.cols(), 0.0);
    }

    // Update cache: cache = cache + gradient²
    layer.weight_cache = layer.weight_cache + pow(layer.dweights, 2);
    layer.bias_cache = layer.bias_cache + pow(layer.dbiases, 2);

    // Update parameters: weights = weights - learning_rate * gradients / (sqrt(cache) + epsilon)
    layer.weights = layer.weights + (-current_learning_rate * layer.dweights / (sqrt(layer.weight_cache) + eps));
    layer.biases = layer.biases + (-current_learning_rate * layer.dbiases / (sqrt(layer.bias_cache) + eps));

}

void Optimizer_Adagrad::post_update_params() {
    iterations += 1;
}