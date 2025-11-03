#pragma once
#include "layer_dense.hpp"

class Optimizer_SGD {
public:
    Optimizer_SGD(float learning_rate = 1.0);
    ~Optimizer_SGD() = default;

    void update_params(LayerDense layer);

    float learning_rate;
};
