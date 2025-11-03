#pragma once

#include "activation_softmax.hpp"
#include "categorical_cross_entropy.hpp"
#include "flat_matrix.hpp"
#include <vector>

// Softmax classifier - combined Softmax activation
// and cross-entropy loss for faster backward step
class ActivationSoftmaxLossCategoricalCrossEntropy {
public:
  ActivationSoftmaxLossCategoricalCrossEntropy();
  ~ActivationSoftmaxLossCategoricalCrossEntropy() = default;

  // Forward pass
  double forward(const FlatMatrix &inputs, const std::vector<int> &y_true);
  double forward(const FlatMatrix &inputs, const FlatMatrix &y_true_onehot);
  
  // Backward pass
  void backward(const FlatMatrix &dvalues, const std::vector<int> &y_true);
  void backward(const FlatMatrix &dvalues, const FlatMatrix &y_true_onehot);

  FlatMatrix output;
  FlatMatrix dinputs;

private:
  ActivationSoftmax activation;
  LossCategoricalCrossEntropy loss;
};
