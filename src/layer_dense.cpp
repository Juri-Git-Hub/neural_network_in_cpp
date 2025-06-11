
#include "layer_dense.hpp"
#include "flat_matrix.hpp"
#include "utils.hpp"
#include <stdexcept>
#include <vector>

LayerDense::LayerDense(int n_inputs, int n_neurons)
    : weights(randn_matrix(n_inputs, n_neurons, 0.0, 0.01)),
      biases(std::vector<double>(n_neurons, 0.0)), inputs(0, n_inputs),
      output(0, n_neurons), dinputs(0, n_inputs), dweights(0, n_neurons),
      dbiases(n_neurons, 0.0) {}

void LayerDense::forward(const FlatMatrix &Inputs) {
  if (Inputs.cols() != weights.rows()) {
    throw std::invalid_argument(
        "LayerDense forward: input.cols and weights.rows have to match!");
  }

  this->inputs = Inputs;

  output = matmul(inputs, weights);

  for (int i = 0; i < output.rows(); ++i) {
    for (int j = 0; j < output.cols(); ++j) {
      double currentValue = output.get(i, j);
      output.set(i, j, currentValue + biases[j]);
    }
  }
}

void LayerDense::backward(const FlatMatrix &dvalues) {
  if (dvalues.cols() != weights.cols()) {
    throw std::invalid_argument(
        "LayerDense backward: dvalues.cols and weights.cols have to match!");
  } else if (dvalues.rows() != inputs.rows()) {
    throw std::invalid_argument(
        "LayerDense backward: dvalues.rows and inputs.rows have to match!");
  }

  FlatMatrix Tinputs = transpose(inputs);

  dweights = matmul(Tinputs, dvalues);
  dbiases = sum_cols(dvalues);

  FlatMatrix Tweights = transpose(weights);
  dinputs = matmul(dvalues, Tweights);
}