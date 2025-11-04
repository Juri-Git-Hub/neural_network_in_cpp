#pragma once

#include "flat_matrix.hpp"
#include <vector>

class LayerDense {
public:
  LayerDense(int n_inputs, int n_neurons);
  ~LayerDense() = default;

  void forward(const FlatMatrix &Inputs);
  void backward(const FlatMatrix &dvalues);

  FlatMatrix output;
  FlatMatrix dinputs;
  FlatMatrix weights;
  FlatMatrix biases;
  FlatMatrix dweights;
  FlatMatrix dbiases;
  FlatMatrix weight_momentums;
  FlatMatrix biases_momentum;
  FlatMatrix weight_cache;
  FlatMatrix bias_cache;

private:
  FlatMatrix inputs;
};