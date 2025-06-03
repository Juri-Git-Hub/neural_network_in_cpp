#include "../include/flat_matrix.hpp"
#include "../include/utils.hpp"
#include <iostream>

int main() {
  // 1) Test elementwise_max (ähnlich einer ReLU mit threshold=0.0)
  FlatMatrix M(3, 3, -1.0);
  M.set(0, 0, -2.5);
  M.set(1, 1, 0.5);
  M.set(2, 2, -0.2);
  double threshold = 0.0;

  FlatMatrix R = elementwise_max(M, threshold);
  std::cout << "elementwise_max(M, 0.0) ergibt:\n";
  for (int i = 0; i < R.rows(); ++i) {
    for (int j = 0; j < R.cols(); ++j) {
      std::cout << R.get(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // 2) Test elementwise_mul
  FlatMatrix A(2, 2, 2.0);
  FlatMatrix B(2, 2, 3.0);
  B.set(1, 1, 4.0); // B = [[3,3],[3,4]]
  FlatMatrix C = elementwise_mul(A, B);
  std::cout << "elementwise_mul(A, B) ergibt:\n";
  for (int i = 0; i < C.rows(); ++i) {
    for (int j = 0; j < C.cols(); ++j) {
      std::cout << C.get(i, j) << " ";
    }
    std::cout << "\n";
  }
  return 0;
}
