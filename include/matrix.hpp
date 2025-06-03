#pragma once

#include <vector>

class Matrix {
public:
  Matrix(int rows, int cols, double initVal = 0.0);

  double get(int i, int j) const;

  void set(int i, int j, double value);

  int rows() const;
  int cols() const;

  Matrix operator+(const Matrix &other) const;

private:
  int m_rows;
  int m_cols;
  std::vector<std::vector<double>> m_data;
};