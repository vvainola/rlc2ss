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
#include <unordered_map>

template <class state_vector_t,
          class jacobian_matrix_t,
          class input_matrix_t>
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
    state_vector_t stepBackwardEuler(System const& system, state_vector_t const& x0, double t, double dt);
    template <class InputVector>
    state_vector_t stepLinearBackwardEuler(state_vector_t const& x0, InputVector const& u, double dt);
    template <class System>
    state_vector_t stepTustin(System const& system, state_vector_t const& x0, double t, double dt);
    template <class InputVector>
    state_vector_t stepLinearTustin(state_vector_t const& x0, InputVector const& u, double dt);

    void updateJacobian(jacobian_matrix_t const& jacobian) {
        m_dt_prev = -1;
        m_jacobian = jacobian;
        uint64_t hash = matrixHash(jacobian);
        m_used_euler_cache = &m_euler_caches[hash];
        m_used_tustin_cache = &m_tustin_caches[hash];
        m_used_linear_euler_cache = &m_linear_euler_caches[hash];
        m_used_linear_tustin_cache = &m_linear_tustin_caches[hash];
    }

    void updateSystem(jacobian_matrix_t const& jacobian, input_matrix_t const& B) {
        m_dt_prev = -1;
        m_jacobian = jacobian;
        m_input_matrix = B;
        uint64_t hash = matrixHash(jacobian);
        uint64_t input_hash = matrixHash(B);
        hash ^= input_hash + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        m_used_euler_cache = &m_euler_caches[hash];
        m_used_tustin_cache = &m_tustin_caches[hash];
        m_used_linear_euler_cache = &m_linear_euler_caches[hash];
        m_used_linear_tustin_cache = &m_linear_tustin_caches[hash];
    }

    void enableInverseMatrixCaching(bool enable) {
        m_caching_enabled = enable;
    }

    bool initialized() const {
        return m_used_euler_cache != nullptr && m_used_tustin_cache != nullptr && m_used_linear_euler_cache != nullptr && m_used_linear_tustin_cache != nullptr;
    }

  private:
    using MatrixHash = uint64_t;
    struct LinearStepCoefficients {
        jacobian_matrix_t state;
        input_matrix_t input;
    };

    template <class Matrix>
    MatrixHash matrixHash(Matrix const& matrix);

    jacobian_matrix_t m_jacobian;
    input_matrix_t m_input_matrix;
    jacobian_matrix_t* m_jacobian_coeff_inv; // 1 / (1 - 0.5 * dt * J)
    double m_dt_prev = -1;
    double m_epsilon = 1e-8;
    double m_abstol = 1e-6;
    double m_reltol = 1e-3;
    size_t m_max_iterations = 10;

    // Matrx inverse caches
    std::unordered_map<MatrixHash, std::unordered_map<double, jacobian_matrix_t>> m_euler_caches;
    std::unordered_map<MatrixHash, std::unordered_map<double, jacobian_matrix_t>> m_tustin_caches;
    std::unordered_map<MatrixHash, std::unordered_map<double, LinearStepCoefficients>> m_linear_euler_caches;
    std::unordered_map<MatrixHash, std::unordered_map<double, LinearStepCoefficients>> m_linear_tustin_caches;
    std::unordered_map<double, jacobian_matrix_t>* m_used_euler_cache = nullptr;
    std::unordered_map<double, jacobian_matrix_t>* m_used_tustin_cache = nullptr;
    std::unordered_map<double, LinearStepCoefficients>* m_used_linear_euler_cache = nullptr;
    std::unordered_map<double, LinearStepCoefficients>* m_used_linear_tustin_cache = nullptr;
    bool m_caching_enabled = false;

    bool withinTolerances(state_vector_t const& x, state_vector_t const& err) {
        for (int i = 0; i < x.size(); ++i) {
            if (abs(err[i]) > std::max(m_reltol * abs(x[i]), m_abstol)) {
                return false;
            }
        }
        return true;
    }
};

template <class state_vector_t,
          class jacobian_matrix_t,
          class input_matrix_t>
template <class System>
inline state_vector_t Integrator<state_vector_t, jacobian_matrix_t, input_matrix_t>::stepBackwardEuler(System const& system, state_vector_t const& x0, double t, double dt) {
    t += dt;
    jacobian_matrix_t* jacobian_coeff_inv;
    if (!m_caching_enabled) {
        m_used_euler_cache->clear();
    }
    auto it = m_used_euler_cache->find(dt);
    if (it == m_used_euler_cache->end()) {
        jacobian_coeff_inv = &m_used_euler_cache->emplace(dt, (jacobian_matrix_t::Identity() - dt * m_jacobian).inverse()).first->second;
    } else {
        jacobian_coeff_inv = &it->second;
    }

    // apply first Newton step
    state_vector_t dxdt = system.dxdt(x0, t);
    state_vector_t diff = *jacobian_coeff_inv * (-dt * dxdt);
    state_vector_t x = x0 - diff;

    // iterate Newton until some precision is reached
    size_t iterations = 0;
    while (diff.norm() > m_epsilon && iterations < m_max_iterations) {
        dxdt = system.dxdt(x, t);
        diff = *jacobian_coeff_inv * (x - x0 - dt * dxdt);
        x -= diff;
        ++iterations;
    }
    return x;
}

template <class state_vector_t,
          class jacobian_matrix_t,
          class input_matrix_t>
template <class InputVector>
inline state_vector_t Integrator<state_vector_t, jacobian_matrix_t, input_matrix_t>::stepLinearBackwardEuler(
    state_vector_t const& x0, InputVector const& u, double dt) {
    // Cache the complete discrete input matrix
    //
    //     Bd = inv(I - dt*A) * dt * B
    //
    // so each step only evaluates Ad*x + Bd*u.
    if (!m_caching_enabled) {
        m_used_linear_euler_cache->clear();
    }

    auto it = m_used_linear_euler_cache->find(dt);
    if (it == m_used_linear_euler_cache->end()) {
        jacobian_matrix_t lhs_inv = (jacobian_matrix_t::Identity() - dt * m_jacobian).inverse();
        LinearStepCoefficients coeffs{
            .state = lhs_inv,
            .input = lhs_inv * (dt * m_input_matrix),
        };
        it = m_used_linear_euler_cache->emplace(dt, std::move(coeffs)).first;
    }

    LinearStepCoefficients const& coeffs = it->second;
    return coeffs.state * x0 + coeffs.input * u;
}

template <class state_vector_t,
          class jacobian_matrix_t,
          class input_matrix_t>
template <class System>
inline state_vector_t Integrator<state_vector_t, jacobian_matrix_t, input_matrix_t>::stepTustin(System const& system, state_vector_t const& x0, double t, double dt) {
    if (dt != m_dt_prev) {
        m_dt_prev = dt;
        if (!m_caching_enabled) {
            m_used_tustin_cache->clear();
        }
        // Update 1 / (1 - 0.5 * dt * J) term
        auto it = m_used_tustin_cache->find(dt);
        if (it == m_used_tustin_cache->end()) {
            m_jacobian_coeff_inv = &m_used_tustin_cache->emplace(dt, (jacobian_matrix_t::Identity() - 0.5 * dt * m_jacobian).inverse()).first->second;
        } else {
            m_jacobian_coeff_inv = &it->second;
        }
    }

    // apply first Newton step
    state_vector_t dxdt0 = system.dxdt(x0, t);
    state_vector_t diff = *m_jacobian_coeff_inv * (-0.5 * dt * dxdt0);
    state_vector_t x = x0 - diff;
    t += dt;

    // iterate Newton until some precision is reached
    size_t iterations = 0;
    while (diff.norm() > m_epsilon && iterations < m_max_iterations) {
        state_vector_t dxdt = system.dxdt(x, t);
        diff = *m_jacobian_coeff_inv * (x - x0 - 0.5 * dt * (dxdt0 + dxdt));
        x -= diff;
        ++iterations;
    }
    return x;
}

template <class state_vector_t,
          class jacobian_matrix_t,
          class input_matrix_t>
template <class InputVector>
inline state_vector_t Integrator<state_vector_t, jacobian_matrix_t, input_matrix_t>::stepLinearTustin(
    state_vector_t const& x0, InputVector const& u, double dt) {
    // Cache the complete discrete input matrix
    //
    //     Bd = inv(I - 0.5*dt*A) * dt * B
    //
    // so each step only evaluates Ad*x + Bd*u.
    if (!m_caching_enabled) {
        m_used_linear_tustin_cache->clear();
    }

    auto it = m_used_linear_tustin_cache->find(dt);
    if (it == m_used_linear_tustin_cache->end()) {
        jacobian_matrix_t lhs_inv = (jacobian_matrix_t::Identity() - 0.5 * dt * m_jacobian).inverse();
        LinearStepCoefficients coeffs{
            .state = lhs_inv * (jacobian_matrix_t::Identity() + 0.5 * dt * m_jacobian),
            .input = lhs_inv * (dt * m_input_matrix),
        };
        it = m_used_linear_tustin_cache->emplace(dt, std::move(coeffs)).first;
    }

    LinearStepCoefficients const& coeffs = it->second;
    return coeffs.state * x0 + coeffs.input * u;
}

template <class state_vector_t, class jacobian_matrix_t, class input_matrix_t>
template <class Matrix>
inline uint64_t Integrator<state_vector_t, jacobian_matrix_t, input_matrix_t>::matrixHash(Matrix const& matrix) {
    // Hash function for Eigen matrix and vector.
    // The code is from `hash_combine` function of the Boost library. See
    // http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
    // https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
    // Note that it is oblivious to the storage order of Eigen matrix (column- or
    // row-major). It will give you the same hash value for two different matrices if they
    // are the transpose of each other in different storage order.
    uint64_t seed = 0;
    for (int i = 0; i < matrix.size(); ++i) {
        auto elem = *(matrix.data() + i);
        seed ^= std::hash<typename Matrix::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
