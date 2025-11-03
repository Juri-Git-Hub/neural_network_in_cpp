#include "../include/activation_softmax_loss_categorical_cross_entropy.hpp"
#include <stdexcept>

ActivationSoftmaxLossCategoricalCrossEntropy::ActivationSoftmaxLossCategoricalCrossEntropy() 
    : activation(), loss() {
}

// Forward pass with sparse labels
double ActivationSoftmaxLossCategoricalCrossEntropy::forward(const FlatMatrix &inputs, const std::vector<int> &y_true) {
  // Output layer's activation function
  activation.forward(inputs);
  
  // Set the output
  output = activation.output;
  
  // Calculate and return loss value
  return loss.forward(output, y_true);
}

// Forward pass with one-hot encoded labels
double ActivationSoftmaxLossCategoricalCrossEntropy::forward(const FlatMatrix &inputs, const FlatMatrix &y_true_onehot) {
  // Output layer's activation function
  activation.forward(inputs);
  
  // Set the output
  output = activation.output;
  
  // Calculate and return loss value
  return loss.forward(output, y_true_onehot);
}

// Backward pass with sparse labels
void ActivationSoftmaxLossCategoricalCrossEntropy::backward(const FlatMatrix &dvalues, const std::vector<int> &y_true) {
  int samples = dvalues.rows();
  int labels = dvalues.cols();
  
  if (samples != y_true.size())
  {
    throw std::invalid_argument{"ActivationSoftmaxLossCCE::backward: number of labels doesn't match samples!"};
  }
  
  // Copy dvalues so we can safely modify
  dinputs = dvalues;
  
  // Calculate gradient: dvalues[range(samples), y_true] -= 1
  for (int i = 0; i < samples; ++i)
  {
    int true_label = y_true[i];
    double current = dinputs.get(i, true_label);
    dinputs.set(i, true_label, current - 1.0);
  }
  
  // Normalize gradient
  for (int i = 0; i < samples; ++i)
  {
    for (int j = 0; j < labels; ++j)
    {
      double normalized = dinputs.get(i, j) / samples;
      dinputs.set(i, j, normalized);
    }
  }
}

// Backward pass with one-hot encoded labels
void ActivationSoftmaxLossCategoricalCrossEntropy::backward(const FlatMatrix &dvalues, const FlatMatrix &y_true_onehot) {
  int samples = dvalues.rows();
  
  // If labels are one-hot encoded, turn them into discrete values
  std::vector<int> y_true_discrete(samples);
  for (int i = 0; i < samples; ++i)
  {
    int max_idx = 0;
    double max_val = y_true_onehot.get(i, 0);
    
    for (int j = 1; j < y_true_onehot.cols(); ++j)
    {
      double val = y_true_onehot.get(i, j);
      if (val > max_val)
      {
        max_val = val;
        max_idx = j;
      }
    }
    y_true_discrete[i] = max_idx;
  }
  
  // Use the sparse label backward pass
  backward(dvalues, y_true_discrete);
}
