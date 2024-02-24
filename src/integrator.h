// MIT License
//
// Copyright (c) 2022 vvainola
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Implicit integration using backward euler / Tustin. Backward euler from
// boost odeint was used as an example

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <queue>
#include <map>

template <class vector_t,
          class matrix_t>
class Integrator {
  public:
    Integrator()
        : m_dt_prev(0) {
    }

    void setTolerances(double epsilon, double abstol, double reltol) {
        m_epsilon = epsilon;
        m_abstol = abstol;
        m_reltol = reltol;
    }

    // <summary>
    /// Adaptive integration using Runge-Kutta-Fehlberg method https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    /// <param name="system">System with dxdt and jacobian functions</param>
    /// <param name="x0">Initial state</param>
    /// <param name="t">Current time</param>
    /// <returns>New state</returns>
    template <class System>
    vector_t stepRungeKuttaFehlberg(System const& system, vector_t const& x0, double t, double dt);

    /// <summary>
    /// Do a step with backward euler integration. The next step is solved with
    /// Newton's method
    ///                y(k) - y(0) - dt * f(t + dt, y(k))
    // y(k+1) = y(k) - ----------------------------------
    //                      1 - dt * J(t + dt, y(k)
    /// </summary>
    /// <param name="system">System with dxdt and jacobian functions</param>
    /// <param name="x0">Initial state</param>
    /// <param name="t">Current time</param>
    /// <returns>New state</returns>
    template <class System>
    vector_t stepBackwardEuler(System const& system, vector_t const& x0, double t, double dt);
    template <class System>
    vector_t stepTustin(System const& system, vector_t const& x0, double t, double dt);

    void updateJacobian(matrix_t const& jacobian) {
        m_dt_prev = 0;
        m_jacobian = jacobian;
        if (m_caching_enabled) {
            m_jacobian_hash = matrixHash(jacobian);
        } else {
            m_jacobian_hash = 0;
        }
    }

    void enableInverseMatrixCaching(bool enable) {
        m_caching_enabled = enable;
    }

  private:
    uint64_t matrixHash(matrix_t const& matrix);

    matrix_t m_jacobian;
    uint64_t m_jacobian_hash;
    matrix_t* m_jacobian_coeff_inv; // 1 / (1 - 0.5 * dt * J)
    double m_dt_prev = 0;
    double m_epsilon = 1e-8;
    double m_abstol = 1e-6;
    double m_reltol = 1e-3;
    size_t m_max_iterations = 10;

    std::map<std::pair<uint64_t, double>, matrix_t> m_inverse_cache;
    uint64_t m_matrix_hash;
    bool m_caching_enabled;

    bool withinTolerances(vector_t const& x, vector_t const& err) {
        for (int i = 0; i < x.size(); ++i) {
            if (abs(err[i]) > std::max(m_reltol * abs(x[i]), m_abstol)) {
                return false;
            }
        }
        return true;
    }
};

template <class vector_t,
          class matrix_t>
template <class System>
inline vector_t Integrator<vector_t, matrix_t>::stepRungeKuttaFehlberg(System const& system, vector_t const& x0, double t, double dt) {
    vector_t x = x0;
    int steps_remaining = 1;
    while (steps_remaining > 0) {
        // clang-format off
        vector_t k1 = dt  * system.dxdt(x                                                                                                           , t);
        vector_t k2 = dt  * system.dxdt(x + 1. / 4.        * k1                                                                                     , t + 1. / 4.    * dt);
        vector_t k3 = dt  * system.dxdt(x + 3. / 32.       * k1 + 9. / 32.       * k2                                                               , t + 3. / 8.    * dt);
        vector_t k4 = dt  * system.dxdt(x + 1932. / 2197.  * k1 - 7200. / 2197.  * k2 + 7296. / 2197.  * k3                                         , t + 12. / 13.  * dt);
        vector_t k5 = dt  * system.dxdt(x + 439. / 216.    * k1 - 8.             * k2 + 3680. / 513.   * k3 - 845. / 4104.   * k4                   , t + dt);
        vector_t k6 = dt  * system.dxdt(x + -8. / 27.      * k1 + 2.             * k2 - 3544. / 2565.  * k3 + 1859. / 4104.  * k4 - 11. / 40.  * k5 , t + 1. / 2.  * dt);
        // clang-format on
        vector_t b5_1 = 16. / 135. * k1;
        vector_t b5_3 = 6656. / 12825. * k3;
        vector_t b5_4 = 28561. / 56430. * k4;
        vector_t b5_5 = -9. / 50. * k5;
        vector_t b5_6 = 2. / 55. * k6;

        vector_t b4_1 = 25. / 216. * k1;
        vector_t b4_3 = 1408. / 2565. * k3;
        vector_t b4_4 = 2197. / 4104. * k4;
        vector_t b4_5 = -1. / 5. * k5;

        vector_t err = ((b5_1 - b4_1) + (b5_3 - b4_3) + (b5_4 - b4_4) + (b5_5 - b4_5) + (b5_6));
        vector_t order_5 = x + b5_1 + b5_3 + b5_4 + b5_5 + b5_6;
        if (withinTolerances(order_5, err)) {
            t += dt;
            x = order_5;
            --steps_remaining;
        } else {
            steps_remaining *= 2;
            dt = dt / 2.;
        }
    }
    return x;
}

template <class vector_t,
          class matrix_t>
template <class System>
inline vector_t Integrator<vector_t, matrix_t>::stepBackwardEuler(System const& system, vector_t const& x0, double t, double dt) {
    t += dt;
    matrix_t jacobian_coeff_inv = (matrix_t::Identity() - dt * m_jacobian).inverse();

    // apply first Newton step
    vector_t dxdt = system.dxdt(x0, t);
    vector_t diff = jacobian_coeff_inv * (-dt * dxdt);
    vector_t x = x0 - diff;

    // iterate Newton until some precision is reached
    size_t iterations = 0;
    while (diff.norm() > m_epsilon && iterations < m_max_iterations) {
        dxdt = system.dxdt(x, t);
        diff = jacobian_coeff_inv * (x - x0 - dt * dxdt);
        x -= diff;
        ++iterations;
    }
    return x;
}

template <class vector_t,
          class matrix_t>
template <class System>
inline vector_t Integrator<vector_t, matrix_t>::stepTustin(System const& system, vector_t const& x0, double t, double dt) {
    t += dt;
    if (dt != m_dt_prev) {
        if (!m_caching_enabled) {
            m_inverse_cache.clear();
        }
        // Update 1 / (1 - 0.5 * dt * J) term
        m_dt_prev = dt;
        std::pair<uint64_t, double> hash_and_dt{m_jacobian_hash, dt};
        if (!m_inverse_cache.contains(hash_and_dt)) {
            m_inverse_cache[hash_and_dt] = (matrix_t::Identity() - 0.5 * dt * m_jacobian).inverse();
        }
        m_jacobian_coeff_inv = &m_inverse_cache[hash_and_dt];
    }

    // apply first Newton step
    vector_t dxdt0 = system.dxdt(x0, t);
    vector_t diff = *m_jacobian_coeff_inv * (-0.5 * dt * dxdt0);
    vector_t x = x0 - diff;

    // iterate Newton until some precision is reached
    size_t iterations = 0;
    while (diff.norm() > m_epsilon && iterations < m_max_iterations) {
        vector_t dxdt = system.dxdt(x, t);
        diff = *m_jacobian_coeff_inv * (x - x0 - 0.5 * dt * (dxdt0 + dxdt));
        x -= diff;
        ++iterations;
    }
    return x;
}

template <class vector_t, class matrix_t>
inline uint64_t Integrator<vector_t, matrix_t>::matrixHash(matrix_t const& matrix) {
    // Hash function for Eigen matrix and vector.
    // The code is from `hash_combine` function of the Boost library. See
    // http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
    // https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
    // Note that it is oblivious to the storage order of Eigen matrix (column- or
    // row-major). It will give you the same hash value for two different matrices if they
    // are the transpose of each other in different storage order.
    uint64_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
        auto elem = *(matrix.data() + i);
        seed ^= std::hash<typename matrix_t::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
