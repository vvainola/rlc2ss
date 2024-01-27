// MIT License
//
// Copyright (c) 2023 vvainola
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include <cmath>

inline const float SQRT3 = float(sqrt(3.0));
inline const float PI = 3.14159265359f;
inline const float PI2 = 6.28318530718f;

#pragma warning(disable : 4201)

template <typename Derived, unsigned SIZE, typename T>
class VecBase {
  public:
    Derived operator+(Derived const& other) const{
        Derived* this_ = static_cast<Derived*>(this);
        Derived temp;
        for (unsigned i = 0; i < SIZE; ++i) {
            temp.data[i] = this_->data[i] + other.data[i];
        }
        return temp;
    }

    Derived& operator+=(Derived const& other) {
        Derived* this_ = static_cast<Derived*>(this);
        for (unsigned i = 0; i < SIZE; ++i) {
            this_->data[i] += other.data[i];
        }
        return *this_;
    }

    Derived operator-(Derived const& other) const {
        Derived* this_ = static_cast<Derived*>(this);
        Derived temp;
        for (unsigned i = 0; i < SIZE; ++i) {
            temp.data[i] = this_->data[i] - other.data[i];
        }
        return temp;
    }

    Derived& operator-=(Derived const& other) {
        Derived* this_ = static_cast<Derived*>(this);
        for (unsigned i = 0; i < SIZE; ++i) {
            this_->data[i] -= other.data[i];
        }
        return *this_;
    }

    template <typename scalar_t>
    Derived operator*(scalar_t scalar) const {
        Derived* this_ = static_cast<Derived*>(this);
        Derived temp;
        for (unsigned i = 0; i < SIZE; ++i) {
            temp.data[i] = this_->data[i] * scalar;
        }
        return temp;
    }

    template <typename scalar_t>
    Derived& operator*=(scalar_t scalar) {
        Derived* this_ = static_cast<Derived*>(this);
        for (unsigned i = 0; i < SIZE; ++i) {
            this_->data[i] *= scalar;
        }
        return *this_;
    }

    template <typename scalar_t>
    Derived operator/(scalar_t scalar) const {
        Derived* this_ = static_cast<Derived*>(this);
        Derived temp;
        T inv = T(1.0) / scalar;
        for (unsigned i = 0; i < SIZE; ++i) {
            temp.data[i] = this_->data[i] * inv;
        }
        return temp;
    }

    template <typename scalar_t>
    Derived& operator/=(scalar_t scalar) {
        Derived* this_ = static_cast<Derived*>(this);
        T inv = T(1.0) / scalar;
        for (unsigned i = 0; i < SIZE; ++i) {
            this_->data[i] *= inv;
        }
        return *this_;
    }
};

template <typename scalar_t, typename Derived, unsigned SIZE, typename T>
Derived operator*(scalar_t scalar, VecBase<Derived, SIZE, T> const& vec) {
    Derived const& vec_ = static_cast<Derived const&>(vec);
    Derived temp;
    for (unsigned i = 0; i < SIZE; ++i) {
        temp.data[i] = vec_.data[i] * scalar;
    }
    return temp;
}

template <typename T>
class VecABC;
template <typename T>
class VecXY;
template <typename T>
class VecXYZ;
template <typename T>
class VecDQ;
template <typename T>
class VectorRotation;
template <typename T>
class RotatingVector;

template <typename T>
class VecABC : public VecBase<VecABC<T>, 3, T> {
  public:
    union {
        struct {
            T a;
            T b;
            T c;
        };
        T data[3];
    };
    VecABC()
        : a(0), b(0), c(0) {}
    VecABC(T init)
        : a(init), b(init), c(init) {}
    VecABC(T a_, T b_, T c_)
        : a(a_), b(b_), c(c_) {}
    VecABC(VecXY<T> const& in)
        : a(in.x),
          b(-0.5 * in.x + sqrt(3.0)/2.0 * in.y),
          c(-0.5 * in.x - sqrt(3.0) / 2.0 * in.y) {}
    T sq() const { return a * a + b * b + c * c; }
    T abs() const { return sqrt(sq()); }
};

template <typename T>
class VecXY : public VecBase<VecXY<T>, 2, T> {
  public:
    union {
        struct {
            T x;
            T y;
        };
        T data[2];
    };
    VecXY()
        : x(0), y(0) {}
    VecXY(T init)
        : x(init), y(init) {}
    VecXY(T x_, T y_)
        : x(x_), y(y_) {}
    VecXY(VecABC<T> const& in)
        : x(2.0 / 3.0 * in.a - 1.0 / 3.0 * in.b - 1.0 / 3.0 * in.c),
          y(SQRT3 / 3.0 * in.b - SQRT3 / 3.0 * in.c) {}
    void transformFromDQ(VecDQ<T> const& in, VectorRotation<T> const& theta) {
        x = in.d * theta.cos - in.q * theta.sin;
        y = in.d * theta.sin + in.q * theta.cos;
    }
    T sq() const { return x * x + y * y; }
    T abs() const { return sqrt(sq()); }

    void rotate(VectorRotation<T> const& theta) {
        T x_tmp = x;
        x = theta.cos * x - theta.sin * y;
        y = theta.cos * y + theta.sin * x_tmp;
    }
};
template <typename T>
static VecXY<T> DQtoXY(VecDQ<T> const& in, VectorRotation<T> const& theta) {
    return VecXY<T>(in.d * theta.cos - in.q * theta.sin, in.d * theta.sin + in.q * theta.cos);
}

template <typename T>
class VecXYZ : public VecBase<VecXYZ<T>, 3, T> {
  public:
    union {
        struct {
            T x;
            T y;
            T z;
        };
        T data[3];
    };
    VecXYZ()
        : x(0), y(0), z(0) {}
    VecXYZ(T init)
        : x(init), y(init), z(init) {}
    VecXYZ(T x_, T y_, T z_)
        : x(x_), y(y_), z(z_) {}
    VecXYZ(VecABC<T> const& in)
        : x(T(2.0 / 3.0) * in.a - T(1.0 / 3.0) * in.b - T(1.0 / 3.0) * in.c),
          y(SQRT3 / 3.0 * in.b - SQRT3 / 3.0 * in.c),
          z(T(1.0 / 3.0) * in.a + T(1.0 / 3.0) * in.b + T(1.0 / 3.0) * in.c) {}
    T sq() const { return x * x + y * y + z * z; }
    T abs() const { return sqrt(sq()); }
};

template <typename T>
class VecDQ : public VecBase<VecDQ<T>, 2, T> {
  public:
    union {
        struct {
            T d;
            T q;
        };
        T data[2];
    };
    VecDQ()
        : d(0), q(0) {}
    VecDQ(T init)
        : d(init), q(init) {}
    VecDQ(T d_, T q_)
        : d(d_), q(q_) {}
    void transformFromXY(VecXY<T> const& in, VectorRotation<T> const& theta) {
        d = in.x * theta.cos + in.y * theta.sin;
        q = -in.x * theta.sin + in.y * theta.cos;
    }
    T sq() const { return d * d + q * q; }
    T abs() const { return sqrt(sq()); }

    void rotate(VectorRotation<T> const& theta) {
        T d_tmp = d;
        d = theta.cos * d - theta.sin * q;
        q = theta.cos * q + theta.sin * d_tmp;
    }
};

template <typename T>
static VecDQ<T> XYtoDQ(VecXY<T> const& in, VectorRotation<T> const& theta) {
    return VecDQ<T>(in.x * theta.cos + in.y * theta.sin, -in.x * theta.sin + in.y * theta.cos);
}

template <typename T>
class VectorRotation {
  public:
    T cos;
    T sin;
    VectorRotation()
        : cos(1), sin(0) {}
    VectorRotation(T cos_, T sin_)
        : cos(cos_), sin(sin_) {}
    VectorRotation(T angle)
        : cos(::cos(angle)), sin(::sin(angle)) {}

    void setRotationAngle(T angle) {
        cos = ::cos(angle);
        sin = ::sin(angle);
    }

    void scaleToUnity() {
        T mag_inv = sqrt(cos * cos + sin * sin);
        cos *= mag_inv;
        sin *= mag_inv;
    }

    void rotate(VectorRotation<T> const& theta) {
        T cos_tmp = cos;
        cos = theta.cos * cos - theta.sin * sin;
        sin = theta.cos * sin + theta.sin * cos_tmp;
    }
};

template <typename T>
class RotatingVector {
  public:
    VectorRotation<T> theta;

    RotatingVector()
        : theta(1, 0),
          m_theta_rot(1, 0) {}
    RotatingVector(T cos_, T sin_, T theta_rot = 0)
        : theta(cos_, sin_),
          m_theta_rot(::cos(theta_rot), ::sin(theta_rot)) {}
    RotatingVector(T theta_rot)
        : theta(1, 0), m_theta_rot(::cos(theta_rot), ::sin(theta_rot)) {}
    void setRotationAngle(T theta_rot) {
        m_theta_rot.x = ::cos(theta_rot);
        m_theta_rot.y = ::sin(theta_rot);
    }

    void rotate() {
        T cos_tmp = theta.cos;
        theta.cos = m_theta_rot.x * theta.cos - m_theta_rot.y * theta.sin;
        theta.sin = m_theta_rot.x * theta.sin + m_theta_rot.y * cos_tmp;
    }

    void scaleToUnity() {
        T mag_inv = sqrt(theta.cos * theta.cos + theta.sin * theta.sin);
        theta.cos *= mag_inv;
        theta.sin *= mag_inv;
    }

  private:
    VecXY<T> m_theta_rot;
};

using V_abc = VecABC<double>;
using Vf_abc = VecABC<float>;
using V_xy = VecXY<double>;
using Vf_xy = VecXY<float>;
using V_xyz = VecXYZ<double>;
using Vf_xyz = VecXYZ<float>;
using V_dq = VecDQ<double>;
using Vf_dq = VecDQ<float>;
