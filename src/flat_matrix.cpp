#include "../include/flat_matrix.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>

int FlatMatrix::index(int i, int j) const {
  if (i < 0 || i >= m_rows || j < 0 || j >= m_cols) {
    throw std::out_of_range("FlatMatrix::index: Index out of range");
  }
  return i * m_cols + j;
}

FlatMatrix::FlatMatrix(int rows, int cols, double initVal)
    : m_rows(rows), m_cols(cols) {
  if (rows < 0 || cols <= 0) {
    throw std::invalid_argument(
        "FlatMatrix: Rows and Columns have to be greater than 0");
  }
  m_data = new double[static_cast<size_t>(rows) * cols];
  for (int i = 0; i < rows * cols; ++i) {
    m_data[i] = initVal;
  }
}

FlatMatrix::FlatMatrix(const FlatMatrix &other)
    : m_rows(other.m_rows), m_cols(other.m_cols) {
  m_data = new double[static_cast<size_t>(m_rows) * m_cols];

  std::copy(other.m_data, other.m_data + static_cast<size_t>(m_rows) * m_cols,
            m_data);
}

double FlatMatrix::get(int i, int j) const {
  int idx = index(i, j);
  return m_data[idx];
}

void FlatMatrix::set(int i, int j, double value) {
  int idx = index(i, j);
  m_data[idx] = value;
}

int FlatMatrix::rows() const { return m_rows; }

int FlatMatrix::cols() const { return m_cols; }

FlatMatrix &FlatMatrix::operator=(const FlatMatrix &other) {
  if (this == &other) {
    return *this;
  }

  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    delete[] m_data;
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_data = new double[static_cast<size_t>(m_rows) * m_cols];
  }

  std::copy(other.m_data, other.m_data + static_cast<size_t>(m_rows) * m_cols,
            m_data);
  return *this;
}

FlatMatrix matmul(const FlatMatrix &A, const FlatMatrix &B) {
  if (A.cols() != B.rows()) {
    throw std::invalid_argument("matmul: cols A and rows B do not match");
  }

  int R = A.rows(); // Rows in the result
  int C = B.cols(); // Columns in the result
  int K = A.cols(); // shared dimensions of matrixes

  FlatMatrix Result(R, C, 0.0);

  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A.get(i, k) * B.get(k, j);
      }
      Result.set(i, j, sum);
    }
  }
  return Result;
}

FlatMatrix::~FlatMatrix() { delete[] m_data; }