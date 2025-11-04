#pragma once

#include "../include/flat_matrix.hpp"
#include <functional>
#include <vector>
#include <cmath>

double randn(double mean = 0.0, double stddev = 1.0);

FlatMatrix randn_matrix(int rows, int cols, double mean, double stddev,
                        double scale = 1.0);

FlatMatrix create_matrix(int rows, int cols, double initVal = 0.0);

FlatMatrix transpose(const FlatMatrix &M);

std::vector<double> sum_rows(const FlatMatrix &M);
std::vector<double> sum_cols(const FlatMatrix &M);
FlatMatrix sum_cols_as_matrix(const FlatMatrix &M);

FlatMatrix elementwise_max(const FlatMatrix &M, double threshold);

FlatMatrix elementwise_mul(const FlatMatrix &A, const FlatMatrix &B);

FlatMatrix add_bias(const FlatMatrix &M, const FlatMatrix &bias);

FlatMatrix softmax_jacobian(const std::vector<double> &p);

double numerical_gradient(std::function<double(const FlatMatrix &)> f,
                          const FlatMatrix &W, int i, int j, double eps = 1e-5);