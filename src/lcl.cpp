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

#include <fstream>
#include <iostream>
#include <sstream>
#include "..\schematics\LCL_matrices.h"
#include <bitset>


struct abc {
    double a;
    double b;
    double c;
};

constexpr double PI = 3.141592652;

class Plant
{
public:

    Plant()
    {
        double Lc_a = 0.1;
        double Lc_b = 0.1;
        double Lc_c = 0.1;
        double Lc = Lc_a;
        double Rc = 1;
        double Rc_a = 1;
        double Rc_b = 1;
        double Rc_c = 1;
        double Rcp_a = 1;
        double Rcp_b = 1;
        double Rcp_c = 1;

        double Rf = 0.1;
        double Rf_a = 0.1;
        double Rf_b = 0.1;
        double Rf_c = 0.1;
        double C = 1e-3;
        double C_a = 1e-3;
        double C_b = 1e-3;
        double C_c = 1e-3;

        double Lg = 1;
        double Lg_a = 1;
        double Lg_b = 1;
        double Lg_c = 1;
        double Rg = 1;
        double Rg_a = 1;
        double Rg_b = 1;
        double Rg_c = 1;
        double Rgp_a = 1;
        double Rgp_b = 1;
        double Rgp_c = 1;

        double Rmcb = 1;
        double R_mcb_a = 1;
        double R_mcb_b = 1;
        double R_mcb_c = 1;

        double Lfault = 1;
        double Lfault_a = 1;
        double Lfault_b = 1;
        double Lfault_c = 1;
        double Rfault = 1;
        double R_fault_a = 1;
        double R_fault_b = 1;
        double R_fault_c = 1;
        double Rfault_neutral_to_ground = 10;

        double Ls = 1;
        double Ls_a = 1;
        double Ls_b = 1;
        double Ls_c = 1;
        double Rs = 1;
        double Rs_a = 1;
        double Rs_b = 1;
        double Rs_c = 1;

        double R = 1;
        double R_a = 1;
        double R_b = 1;
        double R_c = 1;
        double L = 1;
        double L_a = 1;
        double L_b = 1;
        double L_c = 1;

        Components c;
        c.Lc_a = Lc_a;
        c.Lc_b = Lc_b;
        c.Lc_c = Lc_c;
        c.Lg_a = Lg_a;
        c.Lg_b = Lg_b;
        c.Lg_c = Lg_c;
        c.C_a = C_a;
        c.C_b = C_b;
        c.C_c = C_c;
        c.Rc_a = Rc_a;
        c.Rc_b = Rc_b;
        c.Rc_c = Rc_c;
        c.Rf_a = Rf_a;
        c.Rf_b = Rf_b;
        c.Rf_c = Rf_c;
        c.Rg_a = Rg_a;
        c.Rg_b = Rg_b;
        c.Rg_c = Rg_c;
        
        m_ss = calculateStateSpace(c);
    }

    Eigen::Vector<double, int(State::NUM_STATES)> operator()(Eigen::Vector<double, int(State::NUM_STATES)> const& x, double /*t*/)
    {
        return m_ss.A * x + m_ss.B * m_inputs;
    }

    using matrix_t = Eigen::Matrix<double, int(State::NUM_STATES), int(State::NUM_STATES)>;
    matrix_t jacobian(const Eigen::Vector<double, int(State::NUM_STATES)>& /*x*/, const double& /*t*/) const
    {
        return m_ss.A;
    }

    void step(double dt, abc uconv, abc ugrid)
    {
        m_inputs(int(Input::V_a)) = uconv.a;
        m_inputs(int(Input::V_b)) = uconv.b;
        m_inputs(int(Input::V_c)) = uconv.c;
        m_inputs(int(Input::Vs_a)) = ugrid.a;
        m_inputs(int(Input::Vs_b)) = ugrid.b;
        m_inputs(int(Input::Vs_c)) = ugrid.c;
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
    Plant plant;

    double amplitude = 400;

    double freq = 2 * PI * 50;
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
        u_conv.a = amplitude * sin(freq * t);
        u_conv.b = amplitude * sin(freq * t + b_offset);
        u_conv.c = amplitude * sin(freq * t + c_offset);
        u_grid.a = amplitude * sin(freq * t + angle);
        u_grid.b = amplitude * sin(freq * t + angle + b_offset);
        u_grid.c = amplitude * sin(freq * t + angle + c_offset);
        plant.step(t_step, u_conv, u_grid);

        double i_a = plant.m_outputs[int(Output::V3_a)];
        double i_b = plant.m_outputs[int(Output::V3_b)];
        double i_c = plant.m_outputs[int(Output::V3_c)];
        fout << t << "," << i_a << "," << i_b << "," << i_c << "\n";
    }
    fout.close();
    std::cout << "Done!\n"
              << std::endl;
    system("python ..\\scripts\\plot.py");
}
