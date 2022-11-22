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

// Implicit integration using backward euler / trapezoidal. Backward euler from
// boost odeint was used as an example

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>

template <class vector_t,
          class matrix_t>
class Integrator {
  public:
    Integrator(double epsilon = 1e-6)
        : m_epsilon(epsilon),
          m_dt_prev(0) {
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
    vector_t step(System system, vector_t const& x0, double t, double dt) {
        t += dt;
        // Update 1 / (1 - dt * J) term if dt or jacobian has changed
        matrix_t jacobi = system.jacobian(x0, t);
        if (jacobi != m_jacobi_prev
            || dt != m_dt_prev) {
            m_dt_prev = dt;
            m_jacobi_prev = jacobi;
            m_gradient_inv = matrix_t(matrix_t::Identity() - dt * jacobi).inverse();
        }

        // apply first Newton step
        vector_t dxdt = system.dxdt(x0, t);
        vector_t diff = m_gradient_inv * (-dt * dxdt);
        vector_t x = x0 - diff;

        // iterate Newton until some precision is reached
        size_t iterations = 0;
        while (diff.norm() > m_epsilon && iterations < m_max_iterations) {
            dxdt = system.dxdt(x, t);
            diff = m_gradient_inv * (x - x0 - dt * dxdt);
            x -= diff;
            ++iterations;
        }
        return x;
    }

    template <class System>
    vector_t step_trapezoidal(System system, vector_t const& x0, double t, double dt) {
        t += dt;
        // Update 1 / (1 - 0.5 * dt * J) term if dt or jacobian has changed
        matrix_t jacobi = system.jacobian(x0, t);
        if (jacobi != m_jacobi_prev || dt != m_dt_prev) {
            m_dt_prev = dt;
            m_jacobi_prev = jacobi;
            m_gradient_inv = matrix_t(matrix_t::Identity() - 0.5 * dt * jacobi).inverse();
        }

        // apply first Newton step
        vector_t dxdt0 = system.dxdt(x0, t);
        vector_t diff = m_gradient_inv * (-0.5 * dt * dxdt0);
        vector_t x = x0 - diff;

        // iterate Newton until some precision is reached
        size_t iterations = 0;
        while (diff.norm() > m_epsilon && iterations < m_max_iterations) {
            vector_t dxdt = system.dxdt(x, t);
            diff = m_gradient_inv * (x - x0 - 0.5 * dt * (dxdt0 + dxdt));
            x -= diff;
            ++iterations;
        }
        return x;
    }

  private:
    matrix_t m_gradient_inv; // 1 / (1 - dt * J)
    matrix_t m_jacobi_prev;
    double m_dt_prev;
    double m_epsilon;
    size_t m_max_iterations = 10;
};
