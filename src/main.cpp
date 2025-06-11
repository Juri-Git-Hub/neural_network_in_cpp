#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "../include/flat_matrix.hpp"
#include "../include/layer_dense.hpp"

// ---------- Hilfsfunktionen -------------------------------------------
constexpr double EPS = 1e-9;
bool approx(double a, double b, double eps = EPS) {
  return std::fabs(a - b) < eps;
}

void expect_throw(std::function<void()> fn, const char *msg) {
  bool ok = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    ok = true;
  } catch (...) {
  }
  assert(ok && msg);
}

// ---------- Prüfvorgaben (von Hand/NumPy berechnet) -------------------
void reference_values(std::vector<double> &exp_out,
                      std::vector<double> &exp_dweights,
                      std::vector<double> &exp_dbiases,
                      std::vector<double> &exp_dinputs) {
  // Forward-Erwartung  (2×3)  flach in row-major
  exp_out = {0.95, -0.25, 0.0, 1.15, -0.85, 0.2};

  // dweights         (2×3)
  exp_dweights = {0.5, -0.5, 0.0, 3.5, 1.5, -5.0};

  // dbiases          (Länge 3)
  exp_dbiases = {1.5, 0.5, -2.0};

  // dinputs          (2×2)
  exp_dinputs = {0.2, 0.4, 0.25, 0.1};
}

// ---------- Haupt-Test -------------------------------------------------
int main() {
  /* ------------------------------------------------------------------
     1. Konstruktion & manuelles Setzen deterministischer Gewichte
  ------------------------------------------------------------------ */
  LayerDense layer(2, 3);

  // ► feste Gewichte (2×3) – identisch zu den Referenzwerten
  const std::vector<double> W = {0.10, 0.20, -0.10, 0.40, -0.20, 0.00};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      layer.weights.set(i, j, W[i * 3 + j]);

  // ► Biases (3)
  layer.biases = {0.05, -0.05, 0.10};

  // ► Zwei Input-Samples (2×2)
  FlatMatrix X(2, 2);
  X.set(0, 0, 1.0);
  X.set(0, 1, 2.0);
  X.set(1, 0, -1.0);
  X.set(1, 1, 3.0);

  /* ------------------------------------------------------------------
     2. Forward-Pass: Ergebnis & Dimensionen
  ------------------------------------------------------------------ */
  layer.forward(X);
  assert(layer.output.rows() == 2 && layer.output.cols() == 3);

  std::vector<double> exp_out;
  std::vector<double> exp_dw, exp_db, exp_di;
  reference_values(exp_out, exp_dw, exp_db, exp_di);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      assert(approx(layer.output.get(i, j), exp_out[i * 3 + j]));

  /* ------------------------------------------------------------------
     3. Backward-Pass
  ------------------------------------------------------------------ */
  FlatMatrix dvalues(2, 3);
  // Konstant gewählte Gradient-Matrix
  const double DV[6] = {1, 0, -1, 0.5, 0.5, -1};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      dvalues.set(i, j, DV[i * 3 + j]);

  layer.backward(dvalues);

  // -- dweights prüfen (2×3)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      assert(approx(layer.dweights.get(i, j), exp_dw[i * 3 + j]));

  // -- dbiases prüfen (3)
  for (int j = 0; j < 3; ++j)
    assert(approx(layer.dbiases[j], exp_db[j]));

  // -- dinputs prüfen (2×2)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      assert(approx(layer.dinputs.get(i, j), exp_di[i * 2 + j]));

  /* ------------------------------------------------------------------
     4. Fehlerfälle: Dimension-Mismatch
  ------------------------------------------------------------------ */
  FlatMatrix badX(1, 3); // 3 statt 2 Spalten
  expect_throw([&]() { layer.forward(badX); },
               "Kein Exception-Wurf bei cols-Mismatch");

  FlatMatrix badDV(2, 4); // Spalten≠neurons
  expect_throw([&]() { layer.backward(badDV); },
               "Kein Exception-Wurf bei dvalues-Mismatch");

  /* ------------------------------------------------------------------
     5. Ausgabe nur bei Erfolg
  ------------------------------------------------------------------ */
  std::cout << "All LayerDense tests passed ✔️" << std::endl;
  return 0;
}

/*
cmake ..
cmake --build .
./neural_network_cpp
*/
