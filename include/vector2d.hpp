#pragma once

class Vector2D {
public:
  Vector2D();
  Vector2D(double x, double y);

  double getX() const;
  double getY() const;
  void setX(double x);
  void setY(double y);

  double norm() const;
  void scale(double factor);

private:
  double m_x;
  double m_y;
};