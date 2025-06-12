/********************  tests/test_activation_relu.cpp  *******************/
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

#include "../include/activation_relu.hpp"
#include "../include/flat_matrix.hpp"

// ────────── kleine Hilfen ──────────
constexpr double EPS = 1e-12;
inline bool approx(double a, double b, double eps = EPS) {
  return std::fabs(a - b) < eps;
}

void expect_throw(const std::function<void()> &fn, const char *msg) {
  bool ok = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    ok = true;
  } catch (...) {
  }
  assert(ok && msg);
}

// ────────── Tests ──────────────────
int main() {
  /* ---------------------------------------------------------------
     1. Forward-Pass: Richtigkeit & Shape
  --------------------------------------------------------------- */
  FlatMatrix X(1, 5);
  X.set(0, 0, -2.0);
  X.set(0, 1, -1.0);
  X.set(0, 2, 0.0);
  X.set(0, 3, 1.0);
  X.set(0, 4, 2.0);

  ActivationReLU relu;
  relu.forward(X);

  assert(relu.output.rows() == 1 && relu.output.cols() == 5);

  const double exp_out[5] = {0, 0, 0, 1, 2};
  for (int j = 0; j < 5; ++j)
    assert(approx(relu.output.get(0, j), exp_out[j]));

  /* ---------------------------------------------------------------
     2. Backward-Pass: Gradienten korrekt maskieren
  --------------------------------------------------------------- */
  FlatMatrix dvals(1, 5, 10.0); // Upstream-Gradient überall 10
  relu.backward(dvals);

  const double exp_grad[5] = {0, 0, 0, 10, 10};
  for (int j = 0; j < 5; ++j)
    assert(approx(relu.dinputs.get(0, j), exp_grad[j]));

  /* ---------------------------------------------------------------
     3. Fehlerfälle: Dimensions-Mismatch
  --------------------------------------------------------------- */
  FlatMatrix bad_rows(2, 5, 1.0); // rows != inputs.rows()
  expect_throw([&]() { relu.backward(bad_rows); },
               "ReLU backward: rows mismatch nicht erkannt");

  FlatMatrix bad_cols(1, 4, 1.0); // cols != inputs.cols()
  expect_throw([&]() { relu.backward(bad_cols); },
               "ReLU backward: cols mismatch nicht erkannt");

  /* ---------------------------------------------------------------
     4. Rauch-Test mit großer Matrix (Stabilität, keine asserts)
  --------------------------------------------------------------- */
  FlatMatrix bigX(500, 500, -1.0);
  for (int i = 0; i < bigX.rows(); ++i)
    for (int j = 0; j < bigX.cols(); ++j)
      bigX.set(i, j, (i + j) % 7 - 3); // ein bisschen Variation
  relu.forward(bigX);
  relu.backward(bigX); // nur segfault-Test

  std::cout << "All ActivationReLU tests passed ✔️" << std::endl;

  FlatMatrix A; // default
  assert(A.rows() == 0 && A.cols() == 0);

  A = FlatMatrix(2, 3, 1.0); // Zuweisung auf leeres Objekt
  assert(A.get(1, 2) == 1.0);

  FlatMatrix B = A; // Copy-Ctor
  B.set(0, 0, 5.0);
  assert(A.get(0, 0) == 1.0); // deep copy?

  FlatMatrix C; // Copy-Assignment
  C = A;
  assert(C.get(0, 1) == 1.0);
  return 0;
}

/*
cmake ..
cmake --build .
./neural_network_cpp
*/
