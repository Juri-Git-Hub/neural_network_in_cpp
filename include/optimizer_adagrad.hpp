#pragma once
#include "layer_dense.hpp"

class Optimizer_Adagrad {
public:
    Optimizer_Adagrad(float learning_rate = 1.0f, float decay = 0.0f, float eps = 1e-7);
    ~Optimizer_Adagrad() = default;

    void pre_update_params();
    void update_params(LayerDense &layer);
    void post_update_params();

private:
    float learning_rate;
    float current_learning_rate;
    float decay;
    int iterations;
    float eps;
};
