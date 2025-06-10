#include "../include/utils.hpp"
#include "flat_matrix.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>
#include <vector>

static std::mt19937 global_rng{static_cast<unsigned>(
    std::chrono::high_resolution_clock::now().time_since_epoch().count())};

double randn(double mean, double stddev) {
  std::normal_distribution<double> dist(mean, stddev);
  return dist(global_rng);
}

FlatMatrix randn_matrix(int rows, int cols, double mean, double stddev,
                        double scale) {
  FlatMatrix M(rows, cols, 0.0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double val = randn(mean, stddev) * scale;
      M.set(i, j, val);
    }
  }
  return M;
}

FlatMatrix create_matrix(int rows, int cols, double initVal) {
  return FlatMatrix(rows, cols, initVal);
}

FlatMatrix transpose(const FlatMatrix &M) {
  int R = M.rows();
  int C = M.cols();

  FlatMatrix T(C, R, 0.0);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      T.set(j, i, M.get(i, j));
    }
  }
  return T;
}

std::vector<double> sum_rows(const FlatMatrix &M) {
  int R = M.rows();
  int C = M.cols();

  std::vector<double> sums(R, 0.0);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      sums[i] += M.get(i, j);
    }
  }
  return sums;
}

std::vector<double> sum_cols(const FlatMatrix &M) {
  int R = M.rows();
  int C = M.cols();

  std::vector<double> sums(C, 0.0);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      sums[j] += M.get(i, j);
    }
  }
  return sums;
}

FlatMatrix elementwise_max(const FlatMatrix &M, double threshold) {
  int R = M.rows();
  int C = M.cols();

  FlatMatrix T(R, C, 0.0);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      double element_max = std::max(M.get(i, j), threshold);
      T.set(i, j, element_max);
    }
  }
  return T;
}

FlatMatrix elementwise_mul(const FlatMatrix &A, const FlatMatrix &B) {
  if (A.cols() != B.cols() || A.rows() != B.rows()) {
    throw std::invalid_argument("elementwise_mul: cols A and cols B AND rows A "
                                "and Rows B have to match!");
  }

  int R = A.rows();
  int C = A.cols();

  FlatMatrix M(R, C, 0.0);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      double element_mul = A.get(i, j) * B.get(i, j);
      M.set(i, j, element_mul);
    }
  }
  return M;
}

FlatMatrix softmax_jacobian(const std::vector<double> &p) {
  FlatMatrix D(p.size(), p.size(), 0.0);

  int R = D.rows();
  int C = D.cols();
  for (int i = 0; i < R; ++i) {
    D.set(i, i, p[i]);
  }

  int n = static_cast<int>(p.size());
  FlatMatrix P(n, n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      P.set(i, j, p[i] * p[j]);
    }
  }

  FlatMatrix J = subtract(D, P);

  return J;
}

double numerical_gradient(std::function<double(const FlatMatrix &)> f,
                          const FlatMatrix &W, int i, int j, double eps) {
  FlatMatrix W_plus = W;
  FlatMatrix W_minus = W;

  if (i < 0 || i >= W.rows() || j < 0 || j >= W.cols()) {
    throw std::out_of_range("numerical_gradient: ungültiger Index (i,j).");
  }

  double original_value = W.get(i, j);
  W_plus.set(i, j, original_value + eps);
  W_minus.set(i, j, original_value - eps);

  double f_plus = f(W_plus);
  double f_minus = f(W_minus);

  double gradient_approx = (f_plus - f_minus) / (2.0 * eps);
  return gradient_approx;
}