#include "../include/activation_softmax.hpp"
#include "flat_matrix.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

void ActivationSoftmax::forward(const FlatMatrix &inputs_) {
  inputs = inputs_;

  int R = inputs.rows();
  int C = inputs.cols();

  output = FlatMatrix(R, C, 0.0);

  for (int i = 0; i < R; ++i) {
    double maxVal = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < C; ++j) {
      double value = inputs.get(i, j);
      if (value > maxVal) {
        maxVal = value;
      }
    }

    double sumExp = 0;
    for (int j = 0; j < C; ++j) {
      double exponent = std::exp(inputs.get(i, j) - maxVal);
      output.set(i, j, exponent);
      sumExp += exponent;
    }

    for (int j = 0; j < C; ++j) {
      double finExp = output.get(i, j) / sumExp;
      output.set(i, j, finExp);
    }
  }
}

void ActivationSoftmax::backward(const FlatMatrix &dvalues) {
  
  if (dvalues.rows() != output.rows() || dvalues.cols() != output.cols()) {
    throw std::invalid_argument{"ActivationSoftmax::backward: Invalid input! dvalues has to match output."};
  }
  
  int R = dvalues.rows();
  int C = dvalues.cols();

  dinputs = FlatMatrix(R, C, 0.0);

  for (int i = 0; i < R; ++i)
  {
    double dot = 0.0;

    for (int j = 0; j < C; ++j)
    {
      dot += output.get(i, j) * dvalues.get(i, j);
    }

    for (int j = 0; j < C; ++j)
    {
      double new_val = output.get(i, j) * (dvalues.get(i, j) - dot);
      dinputs.set(i, j, new_val);
    }
  }
}
