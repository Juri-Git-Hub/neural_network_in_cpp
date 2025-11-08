#pragma once
#include "layer_dense.hpp"

class Optimizer_Adam {
public:
    Optimizer_Adam(float learning_rate = 0.001f, float decay = 0.0f, float eps = 1e-7, float beta_1 = 0.9f, float beta_2 = 0.999f);
    ~Optimizer_Adam() = default;

    void pre_update_params();
    void update_params(LayerDense &layer);
    void post_update_params();

private:
    float learning_rate;
    float current_learning_rate;
    float decay;
    int iterations;
    float eps;
    float beta_1;
    float beta_2;
};
