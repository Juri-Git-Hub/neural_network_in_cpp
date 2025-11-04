#pragma once

class FlatMatrix {
public:
  FlatMatrix() : m_rows(0), m_cols(0), m_data(nullptr) {}

  FlatMatrix(int rows, int cols, double initVal = 0.0);

  FlatMatrix(const FlatMatrix &other);

  ~FlatMatrix();

  double get(int i, int j) const;

  void set(int i, int j, double value);

  int rows() const;

  int cols() const;

  FlatMatrix &operator=(const FlatMatrix &other);

private:
  int m_rows;
  int m_cols;
  double *m_data;
  int index(int i, int j) const;
};

FlatMatrix matmul(const FlatMatrix &A, const FlatMatrix &B);

FlatMatrix subtract(const FlatMatrix &A, const FlatMatrix &B);

FlatMatrix operator-(const FlatMatrix &A, const FlatMatrix &B);

FlatMatrix add(const FlatMatrix &A, const FlatMatrix &B);

FlatMatrix operator+(const FlatMatrix &A, const FlatMatrix &B);

FlatMatrix operator-(const FlatMatrix &A, const FlatMatrix &B);

FlatMatrix scalar_multiply(const FlatMatrix &M, double scalar);

FlatMatrix operator*(const FlatMatrix &M, double scalar);

FlatMatrix operator*(double scalar, const FlatMatrix &M);