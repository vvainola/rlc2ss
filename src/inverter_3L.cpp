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
#include "dbg_gui_wrapper.h"

#define ASSERT(cond)    \
    if (cond) {         \
        __debugbreak(); \
    }

constexpr double DIODE_ON_THRESHOLD_VOLTAGE = 0.8;
constexpr double PI = 3.141592652;


template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}
struct XY {
    double x;
    double y;
};

struct ABC {
    double a;
    double b;
    double c;
};

XY abc_to_xy(ABC const& in) {
    return XY{
        2.0 / 3.0 * in.a - 1.0 / 3.0 * in.b - 1.0 / 3.0 * in.c,
        sqrt(3) / 3.0 * in.b - sqrt(3) / 3.0 * in.c};
}

ABC xy_to_abc(XY in) {
    return ABC{
        in.x,
        -0.5 * in.x + 0.5 * sqrt(3) * in.y,
        -0.5 * in.x - 0.5 * sqrt(3) * in.y};
}

double v_dc = 500;
class Plant {
  public:
    Plant()
        : m_model(Model_3L::Components{
            .L_conv_a = 100e-5,
            .L_conv_b = 100e-5,
            .L_conv_c = 100e-5,
            .L_dc_src = 1e-5,
            .L_grid_a = 100e-5,
            .L_grid_b = 100e-5,
            .L_grid_c = 100e-5,
            .L_src_a = 100e-5,
            .L_src_b = 100e-5,
            .L_src_c = 100e-5,
            .C_dc_n = 10e-3,
            .C_dc_p = 10e-3,
            .C_f_a = 1e-5,
            .C_f_b = 1e-5,
            .C_f_c = 1e-5,
            .R_conv_a = 1e-3,
            .R_conv_b = 1e-3,
            .R_conv_c = 1e-3,
            .R_dc_n = 1e-6,
            .R_dc_p = 1e-6,
            .R_dc_src = 10,
            .R_f_a = 1e-3,
            .R_f_b = 1e-3,
            .R_f_c = 1e-3,
            .R_grid_a = 1e-3,
            .R_grid_b = 1e-3,
            .R_grid_c = 1e-3,
            .R_src_a = 1e-3,
            .R_src_b = 1e-3,
            .R_src_c = 1e-3,
            .R_dc_n_0 = 1e6
        }) {
    }

    void step(double dt, ABC ugrid) {
        checkDiodes();

        Model_3L::Inputs inputs;
        inputs.V_dc_src = v_dc;
        inputs.V_src_a = ugrid.a;
        inputs.V_src_b = ugrid.b;
        inputs.V_src_c = ugrid.c;
        m_model.step(dt, inputs);
        dc_voltage = m_model.outputs.N_dc_p - m_model.outputs.N_dc_n;
    }

    void checkDiodes() {
        double u_dc = m_model.outputs.N_dc_p - m_model.outputs.N_dc_n;
        Model_3L::Switches switches = m_model.switches;
        // A pos
        if (m_model.outputs.N_conv_a - m_model.outputs.N_dc_p > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_p_a = 1;
        }
        if (m_model.outputs.N_dc_n - m_model.outputs.N_conv_a > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_n_a = 1;
        }
        if (m_model.outputs.I_L_conv_a > 0) {
            m_model.switches.S_n_a = 0;
        }
        if (m_model.outputs.I_L_conv_a < 0) {
            m_model.switches.S_p_a = 0;
        }

        if (m_model.outputs.N_conv_b - m_model.outputs.N_dc_p > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_p_b = 1;
        }
        if (m_model.outputs.N_dc_n - m_model.outputs.N_conv_b > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_n_b = 1;
        }
        if (m_model.outputs.I_L_conv_b > 0) {
            m_model.switches.S_n_b = 0;
        }
        if (m_model.outputs.I_L_conv_b < 0) {
            m_model.switches.S_p_b = 0;
        }

        if (m_model.outputs.N_conv_c - m_model.outputs.N_dc_p > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_p_c = 1;
        }
        if (m_model.outputs.N_dc_n - m_model.outputs.N_conv_c > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_n_c = 1;
        }
        if (m_model.outputs.I_L_conv_c > 0) {
            m_model.switches.S_n_c = 0;
        }
        if (m_model.outputs.I_L_conv_c < 0) {
            m_model.switches.S_p_c = 0;
        }
        ASSERT((m_model.switches.S_p_a && m_model.switches.S_n_a));
        ASSERT((m_model.switches.S_p_b && m_model.switches.S_n_b));
        ASSERT((m_model.switches.S_p_c && m_model.switches.S_n_c));
    }

    Model_3L m_model;
    double dc_voltage;
};

Plant plant;
XY i_conv;
double t = 0;
int main() {

    double amplitude = 400;

    double freq = 2 * PI * 50;
    double b_offset = -2.0 * PI / 3.0;
    double c_offset = -4.0 * PI / 3.0;
    double angle = PI;

    ABC u_grid;
    double t_step = 10e-6;
    std::ofstream fout("temp.csv");

    DbgGui_create(t_step);
    DbgGui_startUpdateLoop();
    for (; t < 0.2; t += t_step) {
        DbgGui_sample();
        u_grid.a = amplitude * sin(freq * t + angle);
        u_grid.b = amplitude * sin(freq * t + angle + b_offset);
        u_grid.c = amplitude * sin(freq * t + angle + c_offset);
        plant.step(t_step, u_grid);

        fout << t << ","
             << plant.m_model.outputs.N_conv_a << ","
             << plant.m_model.outputs.N_conv_b << ","
             << plant.m_model.outputs.N_conv_c << ","
             << plant.m_model.outputs.I_L_conv_a << ","
             << plant.m_model.outputs.I_L_conv_b << ","
             << plant.m_model.outputs.I_L_conv_c << ","
             << plant.m_model.outputs.N_dc_n << ","
             << plant.m_model.outputs.N_dc_p << ","
             << plant.m_model.switches.S_n_a << ","
             << plant.m_model.switches.S_n_b << ","
             << plant.m_model.switches.S_n_c << ","
             << plant.m_model.switches.S_p_a << ","
             << plant.m_model.switches.S_p_b << ","
             << plant.m_model.switches.S_p_c << ","
             << plant.m_model.outputs.N_dc_p - plant.m_model.outputs.N_dc_n << ","
             << "\n";
        i_conv = abc_to_xy({plant.m_model.outputs.I_L_conv_a,
                            plant.m_model.outputs.I_L_conv_b,
                            plant.m_model.outputs.I_L_conv_c});
        fout.flush();
    }
    fout.close();
    std::cout << "Done!\n"
              << std::endl;
}
