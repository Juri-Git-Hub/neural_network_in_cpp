#include "../include/matrix.hpp"
#include <stdexcept>
#include <vector>

Matrix::Matrix(int rows, int cols, double initVal)
    : m_rows(rows), m_cols(cols),
      m_data(rows, std::vector<double>(cols, initVal)) {}

double Matrix::get(int i, int j) const {
  if (i < 0 || i >= m_rows || j < 0 || j >= m_cols) {
    throw std::out_of_range("Matrix::get: Index out of range");
  }
  return m_data[i][j];
}

void Matrix::set(int i, int j, double value) {
  if (i < 0 || i >= m_rows || j < 0 || j >= m_cols) {
    throw std::out_of_range("Matrix::set: Index out of range");
  }
  m_data[i][j] = value;
}

int Matrix::rows() const { return m_rows; }

int Matrix::cols() const { return m_cols; }

Matrix Matrix::operator+(const Matrix &other) const {
  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    throw std::invalid_argument("Matrix::operator+: Dimensions do not match");
  }

  Matrix result(m_rows, m_cols, 0.0);

  for (int i = 0; i < m_rows; ++i) {
    for (int j = 0; j < m_cols; ++j) {
      double sum = this->m_data[i][j] + other.m_data[i][j];
      result.set(i, j, sum);
    }
  }
  return result;
}