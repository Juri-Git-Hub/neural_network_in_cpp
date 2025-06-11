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
  std::vector<double> biases;
  FlatMatrix dweights;
  std::vector<double> dbiases;

private:
  FlatMatrix inputs;
};