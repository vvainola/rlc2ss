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

#define BOOST_ALLOW_DEPRECATED_HEADERS
#include "integrator.h"

#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include "..\schematics\transformer_matrices.h"


struct abc {
    double a;
    double b;
    double c;
};

constexpr double PI = 3.141592652;

class Transformer
{
public:

    Transformer()
    {
        double L1 = 1e-6;
        double R1 = 1e-6;
        double R2 = 5;

        Components c;
        c.L1 = L1;
        c.R1 = R1;
        c.R2 = R2;
        c.E1 = 10;
        c.F1 = 1.0 / c.E1;
        m_ss = calculateStateSpace(c);
    }

    Eigen::Vector<double, int(State::NUM_STATES)> dxdt(Eigen::Vector<double, int(State::NUM_STATES)> const& x, double /*t*/)
    {
        return m_ss.A * x + m_ss.B * m_inputs;
    }

    using matrix_t = Eigen::Matrix<double, int(State::NUM_STATES), int(State::NUM_STATES)>;
    matrix_t jacobian(const Eigen::Vector<double, int(State::NUM_STATES)>& /*x*/, const double& /*t*/) const
    {
        return m_ss.A;
    }

    void step(double dt, double Vs)
    {
        m_inputs(int(Input::Vs)) = Vs;
        m_x = m_solver.step_trapezoidal(*this, m_x, 0.0, dt);

        m_outputs = m_ss.C * m_x + m_ss.D * m_inputs;
    }

    StateSpaceMatrices m_ss;
    Eigen::Vector<double, int(State::NUM_STATES)> m_x;
    Eigen::Vector<double, int(Input::NUM_INPUTS)> m_inputs;
    Eigen::Vector<double, int(Output::NUM_OUTPUTS)> m_outputs;
    Integrator<Eigen::Vector<double, int(State::NUM_STATES)>, matrix_t> m_solver;
};

int main()
{
    Transformer plant;

    double amplitude = 400 /*/ u_base*/;

    double freq = 2 * PI * 50 /*/ w_base*/;
    double b_offset = -2.0 * PI / 3.0;
    double c_offset = -4.0 * PI / 3.0;
    double angle = PI;

    abc u_conv;
    abc u_grid;
    double t_step = 10e-6;
    std::ofstream fout("temp.csv");
    double t = 0;
    for (; t < 0.2; t += t_step)
    {
        double Vs = amplitude * sin(freq * t);
        plant.step(t_step, Vs);

        double i_a = plant.m_outputs[int(Output::I_L1)];
        double i_b = plant.m_outputs[int(Output::I_R2)];
        double i_c = plant.m_outputs[int(Output::V_R2)];
        fout << t << "," << i_a << "," << i_b << "," << i_c << "\n";
    }
    fout.close();
    std::cout << "Done!\n"
              << std::endl;
    system("python ..\\scripts\\plot.py");
}
