#pragma once
#include "layer_dense.hpp"

class Optimizer_RMSprop {
public:
    Optimizer_RMSprop(float learning_rate = 0.001f, float decay = 0.0f, float eps = 1e-7, float rho = 0.9f);
    ~Optimizer_RMSprop() = default;

    void pre_update_params();
    void update_params(LayerDense &layer);
    void post_update_params();

private:
    float learning_rate;
    float current_learning_rate;
    float decay;
    int iterations;
    float eps;
    float rho;
};
