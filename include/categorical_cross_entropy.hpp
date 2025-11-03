#pragma once

#include "flat_matrix.hpp"
#include <vector>

class LossCategoricalCrossEntropy {
public:  
  ~LossCategoricalCrossEntropy() = default;

  double forward(const FlatMatrix &y_pred, const std::vector<int> &y_true_labels);
  double forward(const FlatMatrix &y_pred, const FlatMatrix &y_true_onehot);
  
  void backward(const FlatMatrix &dvalues, const std::vector<int> &y_true_labels);
  void backward(const FlatMatrix &dvalues, const FlatMatrix &y_true_onehot);
  
  FlatMatrix dinputs;
};