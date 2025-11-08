#include "../include/optimizer_adam.hpp"
#include "../include/utils.hpp"

Optimizer_Adam::Optimizer_Adam(float learning_rate, float decay, float eps, float beta_1, float beta_2)
    : learning_rate(learning_rate), current_learning_rate(learning_rate),
      decay(decay), iterations(0), eps(eps), beta_1(beta_1), beta_2(beta_2) {}


void Optimizer_Adam::pre_update_params() {
    // if there is any decay, then the current learning rate gets updated, based on the decay times the iterations
    if (decay > 0)
    {
        current_learning_rate = learning_rate / (1.0f + decay * iterations);
    }
    
}

void Optimizer_Adam::update_params(LayerDense &layer) {

    if (layer.weight_cache.cols() == 0 || layer.weight_cache.rows() == 0)
    {
        layer.weight_cache = create_matrix(layer.weights.rows(), layer.weights.cols(), 0.0f);
        layer.weight_momentums = create_matrix(layer.weights.rows(), layer.weights.cols(), 0.0f);
        layer.bias_cache = create_matrix(1, layer.biases.cols(), 0.0);
        layer.biases_momentum = create_matrix(1, layer.biases.cols(), 0.0);
    }

    // Update momentum with current gradiants
    layer.weight_momentums = beta_1 * layer.weight_momentums + (1 - beta_1) * layer.dweights;
    layer.biases_momentum = beta_1 * layer.biases_momentum + (1 - beta_1) * layer.dbiases;

    // Get corrected momentum
    FlatMatrix weight_momentums_corrected = layer.weight_momentums * (1 / std::pow(1 - beta_1, iterations + 1));
    FlatMatrix bias_momentums_corrected = layer.biases_momentum * (1 / std::pow(1 - beta_1, iterations + 1));

    // Update cache with squared current gradients
    layer.weight_cache = beta_2 * layer.weight_cache + (1 - beta_2) * std::pow(layer.dweights, 2);
    layer.bias_cache = beta_2 * layer.bias_cache + (1 - beta_2) * std::pow(layer.dbiases, 2);

    // Get corrected cache
    FlatMatrix weight_cache_corrected = layer.weight_cache * (1 / std::pow(1 - beta_2, iterations + 1));
    FlatMatrix bias_cache_corrected = layer.bias_cache * (1 / std::pow(1 - beta_2, iterations + 1));

    // Update parameters: weights = weights - learning_rate * gradients / (sqrt(cache) + epsilon)
    layer.weights = layer.weights + (-current_learning_rate * weight_momentums_corrected / (sqrt(weight_cache_corrected) + eps));
    layer.biases = layer.biases + (-current_learning_rate * bias_momentums_corrected / (sqrt(bias_cache_corrected) + eps));

}

void Optimizer_Adam::post_update_params() {
    iterations += 1;
}
