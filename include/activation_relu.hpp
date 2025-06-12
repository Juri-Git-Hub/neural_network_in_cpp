#pragma once

#include "flat_matrix.hpp"

class ActivationReLU {
public:
  ~ActivationReLU() = default;

  void forward(const FlatMatrix &inputs);
  void backward(const FlatMatrix &dvalues);

  FlatMatrix output, dinputs;

private:
  FlatMatrix inputs;
};