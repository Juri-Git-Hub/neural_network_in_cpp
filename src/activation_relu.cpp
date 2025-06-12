#include "../include/activation_relu.hpp"
#include "flat_matrix.hpp"
#include <algorithm>
#include <stdexcept>

void ActivationReLU::forward(const FlatMatrix &inputs) {
  this->inputs = inputs;

  this->output = FlatMatrix(inputs.rows(), inputs.cols(), 0.0);
  for (int i = 0; i < inputs.rows(); ++i) {
    for (int j = 0; j < inputs.cols(); ++j) {
      this->output.set(i, j, std::max(0.0, inputs.get(i, j)));
    }
  }
}

void ActivationReLU::backward(const FlatMatrix &dvalues) {
  if (dvalues.rows() != inputs.rows() || dvalues.cols() != inputs.cols())
    throw std::invalid_argument("ReLU backward: shape mismatch");

  this->dinputs = dvalues;

  for (int i = 0; i < inputs.rows(); ++i) {
    for (int j = 0; j < inputs.cols(); ++j) {
      if (inputs.get(i, j) <= 0.0)
        dinputs.set(i, j, 0.0);
    }
  }
}