#pragma once
#include "layer_dense.hpp"

class Optimizer_SGD {
public:
    Optimizer_SGD(float learning_rate = 1.0f, float decay = 0.0f, float momentum = 0.0f);
    ~Optimizer_SGD() = default;

    void pre_update_params();
    void update_params(LayerDense &layer);
    void post_update_params();

private:
    float learning_rate;
    float current_learning_rate;
    float decay;
    int iterations;
    float momentum;
};
