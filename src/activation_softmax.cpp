#include "../include/activation_softmax.hpp"
#include "flat_matrix.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

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

  // Enumerate outputs and gradients
  for (int i = 0; i < R; ++i)
  {
    // Create Jacobian matrix for this sample
    // jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
    std::vector<std::vector<double>> jacobian(C, std::vector<double>(C, 0.0));
    
    // Build the Jacobian matrix
    for (int j = 0; j < C; ++j)
    {
      double s_j = output.get(i, j);
      for (int k = 0; k < C; ++k)
      {
        double s_k = output.get(i, k);
        if (j == k)
        {
          // Diagonal: s_j * (1 - s_j)
          jacobian[j][k] = s_j * (1.0 - s_j);
        }
        else
        {
          // Off-diagonal: -s_j * s_k
          jacobian[j][k] = -s_j * s_k;
        }
      }
    }
    
    // Calculate sample-wise gradient: jacobian_matrix @ single_dvalues
    for (int j = 0; j < C; ++j)
    {
      double gradient = 0.0;
      for (int k = 0; k < C; ++k)
      {
        gradient += jacobian[j][k] * dvalues.get(i, k);
      }
      dinputs.set(i, j, gradient);
    }
  }
}
