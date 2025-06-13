#pragma once

#include "flat_matrix.hpp"

class ActivationSoftmax {
public:
  ~ActivationSoftmax() = default;

  void forward(const FlatMatrix &inputs_);
  void backward(const FlatMatrix &dvalues);

  FlatMatrix output, dinputs;

private:
  FlatMatrix inputs;
};