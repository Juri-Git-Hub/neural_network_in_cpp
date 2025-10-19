#pragma once

#include "flat_matrix.hpp"
#include <vector>

class LossCategoricalCrossEntropy {
public:  
  ~LossCategoricalCrossEntropy() = default;

  double forward(const FlatMatrix &y_pred, const std::vector<int> &y_true_labels);
  double forward(const FlatMatrix &y_pred, const FlatMatrix &y_true_onehot);  
};