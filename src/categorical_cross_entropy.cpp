#include "../include/categorical_cross_entropy.hpp"
#include <cmath>
#include <stdexcept>

double LossCategoricalCrossEntropy::forward(const FlatMatrix &y_pred, const std::vector<int> &y_true_labels) {
    if (y_pred.rows() != y_true_labels.size())
    {
        throw std::invalid_argument{"LossCCO: the number of labels is not correct!"};
    }
    
    double loss_sum = 0.0;
    int num_samples = y_true_labels.size();

    for (int i = 0; i < num_samples; i++)
    {
        double p = y_pred.get(i, y_true_labels[i]);
        p = std::max(1e-7, std::min(p, 1 - 1e-7));
        loss_sum += - std::log(p);
    }
    
    return loss_sum / static_cast<double>(num_samples);
}

double LossCategoricalCrossEntropy::forward(const FlatMatrix &y_pred, const FlatMatrix &y_true_onehot) {
    if (y_pred.rows() != y_true_onehot.rows() || y_pred.cols() != y_true_onehot.cols())
    {
        throw std::invalid_argument{"LossCCO: the shape of the one-hot labels is not correct!"};
    }

    int R = y_true_onehot.rows();
    int C = y_true_onehot.cols();

    double sum_of_samples = 0.0;
    
    for (int i = 0; i < R; i++)
    {
        double negative_sample_sum = 0.0;

        for (int j = 0; j < C; j++)
        {
            double p = y_pred.get(i, j);
            double clipped_p = std::max(1e-7, std::min(p, 1 - 1e-7));

            negative_sample_sum += y_true_onehot.get(i,j) * std::log(clipped_p);
        }
        
        sum_of_samples += -negative_sample_sum;
    }
    
    return sum_of_samples / static_cast<double>(R);
}

void LossCategoricalCrossEntropy::backward(const FlatMatrix &dvalues, const std::vector<int> &y_true_labels) {
    if (dvalues.rows() != y_true_labels.size())
    {
        throw std::invalid_argument{"LossCCO::backward: the number of labels is not correct!"};
    }
    
    int samples = y_true_labels.size();
    int labels = dvalues.cols();
    
    // Convert sparse labels to one-hot encoding
    FlatMatrix y_true_onehot(samples, labels, 0.0);
    for (int i = 0; i < samples; ++i)
    {
        y_true_onehot.set(i, y_true_labels[i], 1.0);
    }
    
    // Calculate gradient: -y_true / dvalues
    dinputs = FlatMatrix(samples, labels, 0.0);
    for (int i = 0; i < samples; ++i)
    {
        for (int j = 0; j < labels; ++j)
        {
            double y_t = y_true_onehot.get(i, j);
            double dval = dvalues.get(i, j);
            // Clip to prevent division by zero
            dval = std::max(1e-7, std::min(dval, 1 - 1e-7));
            dinputs.set(i, j, -y_t / dval);
        }
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

void LossCategoricalCrossEntropy::backward(const FlatMatrix &dvalues, const FlatMatrix &y_true_onehot) {
    if (dvalues.rows() != y_true_onehot.rows() || dvalues.cols() != y_true_onehot.cols())
    {
        throw std::invalid_argument{"LossCCO::backward: the shape of the one-hot labels is not correct!"};
    }

    int samples = dvalues.rows();
    int labels = dvalues.cols();

    // Calculate gradient: -y_true / dvalues
    dinputs = FlatMatrix(samples, labels, 0.0);
    for (int i = 0; i < samples; ++i)
    {
        for (int j = 0; j < labels; ++j)
        {
            double y_t = y_true_onehot.get(i, j);
            double dval = dvalues.get(i, j);
            // Clip to prevent division by zero
            dval = std::max(1e-7, std::min(dval, 1 - 1e-7));
            dinputs.set(i, j, -y_t / dval);
        }
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