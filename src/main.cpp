#include "../include/flat_matrix.hpp"
#include "../include/utils.hpp"
#include <chrono>
#include <iostream>

int main() {
  // Wähle N (beispielsweise 200 oder 300)
  int N = 200;
  // Erstelle zwei Zufallsmatrizen A und B
  FlatMatrix A = randn_matrix(N, N, 0.0, 1.0, 1.0);
  FlatMatrix B = randn_matrix(N, N, 0.0, 1.0, 1.0);

  // Starte die Zeitmessung
  auto t_start = std::chrono::high_resolution_clock::now();
  FlatMatrix C = matmul(A, B);
  auto t_end = std::chrono::high_resolution_clock::now();

  // Berechne die vergangene Zeit in Sekunden
  std::chrono::duration<double> duration = t_end - t_start;
  std::cout << "matmul für " << N << "×" << N
            << " dauerte: " << duration.count() << " Sekunden\n";

  return 0;
}

/*
cmake ..
cmake --build .
./neural_network_cpp
*/
