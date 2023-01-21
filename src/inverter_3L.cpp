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

#include "../qucs/3L_matrices.h"
#include <fstream>
#include <iostream>
#include <sstream>

#define ASSERT(cond)    \
    if (cond) {         \
        __debugbreak(); \
    }

struct abc {
    double a;
    double b;
    double c;
};

constexpr double DIODE_ON_THRESHOLD_VOLTAGE = 0.8;

double t = 0;

constexpr double PI = 3.141592652;

struct Out {
    double I_L_a;
    double I_L_b;
    double I_L_c;
    double V3_a;
    double V3_b;
    double V3_c;
    double Vdc_n;
    double Vdc_p;
};

template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

class Plant {
  public:
    Plant()
        : m_out{} {
        m_x.setZero();
        m_inputs.setZero();
        setSwitches(0);
    }

    void setSwitches(SwitchPositions switches) {
        m_switches = switches;
        Components c;
        c.L_conv_a = 100e-5;
        c.L_conv_b = 100e-5;
        c.L_conv_c = 100e-5;
        c.L_grid_a = 100e-5;
        c.L_grid_b = 100e-5;
        c.L_grid_c = 100e-5;
        c.L_src_a = 100e-5;
        c.L_src_b = 100e-5;
        c.L_src_c = 100e-5;
        c.C_dc_p = 10e-3;
        c.C_dc_n = 10e-3;
        c.C_f_a = 1e-6;
        c.C_f_b = 1e-6;
        c.C_f_c = 1e-6;
        c.R_conv_a = 1e-3;
        c.R_conv_b = 1e-3;
        c.R_conv_c = 1e-3;
        c.R_grid_a = 1e-3;
        c.R_grid_b = 1e-3;
        c.R_grid_c = 1e-3;
        c.R_f_a = 1e-3;
        c.R_f_b = 1e-3;
        c.R_f_c = 1e-3;
        c.R_src_a = 1e-3;
        c.R_src_b = 1e-3;
        c.R_src_c = 1e-3;
        c.R_dc_p = 1e-6;
        c.R_dc_n = 1e-6;

        m_ss = calculateStateSpace(c, m_switches.to_ullong());
    }

    Eigen::Vector<double, int(State::NUM_STATES)> dxdt(Eigen::Vector<double, int(State::NUM_STATES)> const& x, double /*t*/) {
        return m_ss.A * x + m_ss.B * m_inputs;
    }

    using matrix_t = Eigen::Matrix<double, int(State::NUM_STATES), int(State::NUM_STATES)>;
    matrix_t jacobian(const Eigen::Vector<double, int(State::NUM_STATES)>& /*x*/, const double& /*t*/) const {
        return m_ss.A;
    }

    void step(double dt, abc ugrid) {
        checkDiodes();

        m_inputs(int(Input::I_load)) = 100;
        m_inputs(int(Input::V_src_a)) = ugrid.a;
        m_inputs(int(Input::V_src_b)) = ugrid.b;
        m_inputs(int(Input::V_src_c)) = ugrid.c;
        m_x = m_solver.step(*this, m_x, 0.0, dt);

        m_outputs = m_ss.C * m_x + m_ss.D * m_inputs;
        m_out.I_L_a = m_outputs[int(Output::I_L_conv_a)];
        m_out.I_L_b = m_outputs[int(Output::I_L_conv_b)];
        m_out.I_L_c = m_outputs[int(Output::I_L_conv_c)];

        m_out.V3_a = m_outputs[int(Output::N_conv_a)];
        m_out.V3_b = m_outputs[int(Output::N_conv_b)];
        m_out.V3_c = m_outputs[int(Output::N_conv_c)];

        m_out.Vdc_p = m_outputs[int(Output::N_dc_p)];
        m_out.Vdc_n = m_outputs[int(Output::N_dc_n)];
    }

    StateSpaceMatrices m_ss;
    Eigen::Vector<double, int(State::NUM_STATES)> m_x;
    Eigen::Vector<double, int(Input::NUM_INPUTS)> m_inputs;
    Eigen::Vector<double, int(Output::NUM_OUTPUTS)> m_outputs;
    Integrator<Eigen::Vector<double, int(State::NUM_STATES)>, matrix_t> m_solver;
    SwitchPositions m_switches = 0;
    Out m_out;

    void checkDiodes() {
        SwitchPositions switches = m_switches;
        double u_dc = m_out.Vdc_p - m_out.Vdc_n;
        // A pos
        if (m_out.V3_a - m_out.Vdc_p > DIODE_ON_THRESHOLD_VOLTAGE) {
            switches.set(int(Switch::S_p_a), 1);
        }
        if (m_out.Vdc_n - m_out.V3_a > DIODE_ON_THRESHOLD_VOLTAGE) {
            switches.set(int(Switch::S_n_a), 1);
        }
        if (m_out.I_L_a > 0) {
            switches.set(int(Switch::S_n_a), 0);
        }
        if (m_out.I_L_a < 0) {
            switches.set(int(Switch::S_p_a), 0);
        }

        if (m_out.V3_b - m_out.Vdc_p > DIODE_ON_THRESHOLD_VOLTAGE) {
            switches.set(int(Switch::S_p_b), 1);
        }
        if (m_out.Vdc_n - m_out.V3_b > DIODE_ON_THRESHOLD_VOLTAGE) {
            switches.set(int(Switch::S_n_b), 1);
        }
        if (m_out.I_L_b > 0) {
            switches.set(int(Switch::S_n_b), 0);
        }
        if (m_out.I_L_b < 0) {
            switches.set(int(Switch::S_p_b), 0);
        }

        if (m_out.V3_c - m_out.Vdc_p > DIODE_ON_THRESHOLD_VOLTAGE) {
            switches.set(int(Switch::S_p_c), 1);
        }
        if (m_out.Vdc_n - m_out.V3_c > DIODE_ON_THRESHOLD_VOLTAGE) {
            switches.set(int(Switch::S_n_c), 1);
        }
        if (m_out.I_L_c > 0) {
            switches.set(int(Switch::S_n_c), 0);
        }
        if (m_out.I_L_c < 0) {
            switches.set(int(Switch::S_p_c), 0);
        }
        ASSERT((switches[int(Switch::S_p_a)] && switches[int(Switch::S_n_a)]));
        ASSERT((switches[int(Switch::S_p_b)] && switches[int(Switch::S_n_b)]));
        ASSERT((switches[int(Switch::S_p_c)] && switches[int(Switch::S_n_c)]));

        if (switches != m_switches) {
            setSwitches(switches);
        }
    }
};

int main() {
    Plant plant;

    double amplitude = 400;

    double freq = 2 * PI * 50;
    double b_offset = -2.0 * PI / 3.0;
    double c_offset = -4.0 * PI / 3.0;
    double angle = PI;

    abc u_grid;
    double t_step = 10e-6;
    std::ofstream fout("temp.csv");
    //fout << "time,a,b,c" << "\n";

    /*SwitchPositions switches{};
    switches[int(Switch::S_p_a)] = 1;
    switches[int(Switch::S_n_b)] = 1;
    plant.setSwitches(switches);*/

    for (; t < 0.1; t += t_step) {
        u_grid.a = amplitude * sin(freq * t + angle);
        u_grid.b = amplitude * sin(freq * t + angle + b_offset);
        u_grid.c = amplitude * sin(freq * t + angle + c_offset);
        plant.step(t_step, u_grid);

        /*double i_a = abs(plant.m_outputs[int(Output::Vdc_p)] - plant.m_outputs[int(Output::Vdc_n)]);
        double i_b = plant.m_x[int(State::V_C_dc)];
        double i_c = 0;*/

        double i_a = plant.m_out.V3_a;
        double i_b = plant.m_out.V3_b;
        double i_c = plant.m_out.V3_c;
       fout << t << ","
             << plant.m_out.V3_a << ","
             << plant.m_out.V3_b << ","
             << plant.m_out.V3_c << ","
             << plant.m_out.I_L_a << ","
             << plant.m_out.I_L_b << ","
             << plant.m_out.I_L_c << ","
             << plant.m_out.Vdc_n << ","
             << plant.m_out.Vdc_p << ","
             << plant.m_out.Vdc_p - plant.m_out.Vdc_n << ","
            << "\n";
    }
    fout.close();
    std::cout << "Done!\n" << std::endl;
}
