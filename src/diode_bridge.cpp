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

#include "..\schematics\diode_bridge_matrices.hpp"
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

double timestamp = 0;

constexpr double PI = 3.141592652;

template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

class DiodeBridge {
  public:
    DiodeBridge()
        : m_model(Model_diode_bridge::Components {
              .L_a = 100e-5,
              .L_b = 100e-5,
              .L_c = 100e-5,
              .C_dc = 10e-3,
              .R_a = 1e-3,
              .R_b = 1e-3,
              .R_c = 1e-3,
              .R_dc = 1e-6,
              .R_load = 1
    }){
    }

    void step(double dt, abc ugrid) {

        Model_diode_bridge::Inputs inputs;
        inputs.V_a = ugrid.a;
        inputs.V_b = ugrid.b;
        inputs.V_c = ugrid.c;
        m_model.step(dt, inputs);
        checkDiodes();
    }

    Model_diode_bridge m_model;

    void checkDiodes() {
        double u_dc = m_model.outputs.Vdc_p - m_model.outputs.Vdc_n;
        Model_diode_bridge::Switches switches = m_model.switches;
        // A pos
        if (m_model.outputs.V3_a - m_model.outputs.Vdc_p > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_p_a = 1;
        }
        if (m_model.outputs.Vdc_n - m_model.outputs.V3_a > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_n_a = 1;
        }
        if (m_model.outputs.I_L_a > 0) {
            m_model.switches.S_n_a = 0;
        }
        if (m_model.outputs.I_L_a < 0) {
            m_model.switches.S_p_a = 0;
        }

        if (m_model.outputs.V3_b - m_model.outputs.Vdc_p > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_p_b = 1;
        }
        if (m_model.outputs.Vdc_n - m_model.outputs.V3_b > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_n_b = 1;
        }
        if (m_model.outputs.I_L_b > 0) {
            m_model.switches.S_n_b = 0;
        }
        if (m_model.outputs.I_L_b < 0) {
            m_model.switches.S_p_b = 0;
        }

        if (m_model.outputs.V3_c - m_model.outputs.Vdc_p > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_p_c = 1;
        }
        if (m_model.outputs.Vdc_n - m_model.outputs.V3_c > DIODE_ON_THRESHOLD_VOLTAGE) {
            m_model.switches.S_n_c = 1;
        }
        if (m_model.outputs.I_L_c > 0) {
            m_model.switches.S_n_c = 0;
        }
        if (m_model.outputs.I_L_c < 0) {
            m_model.switches.S_p_c = 0;
        }

        // clang-format off
        // Keep always one switch on so that the DC is connected to ground
        if (m_model.switches.all == 0) {
            if (m_model.switches.all == 0 && switches.S_n_a) m_model.switches.S_n_a = 1;
            if (m_model.switches.all == 0 && switches.S_n_b) m_model.switches.S_n_b = 1;
            if (m_model.switches.all == 0 && switches.S_n_c) m_model.switches.S_n_c = 1;
            if (m_model.switches.all == 0 && switches.S_p_a) m_model.switches.S_p_a = 1;
            if (m_model.switches.all == 0 && switches.S_p_b) m_model.switches.S_p_b = 1;
            if (m_model.switches.all == 0 && switches.S_p_c) m_model.switches.S_p_c = 1;
        }
        // clang-format on
    }
};

int main() {
    DiodeBridge diode_bridge;

    double amplitude = 400;

    double freq = 2 * PI * 50;
    double b_offset = -2.0 * PI / 3.0;
    double c_offset = -4.0 * PI / 3.0;
    double angle = PI;

    abc u_grid;
    double t_step = 10e-6;
    std::ofstream fout("temp.csv");
    fout << "time,a,b,c"
         << "\n";

    for (; timestamp < 0.1; timestamp += t_step) {
        u_grid.a = amplitude * sin(freq * timestamp + angle);
        u_grid.b = amplitude * sin(freq * timestamp + angle + b_offset);
        u_grid.c = amplitude * sin(freq * timestamp + angle + c_offset);
        diode_bridge.step(t_step, u_grid);

        /*double i_a = abs(plant.m_plant.outputs.Vdc_p - plant.m_plant.outputs.Vdc_n);
        double i_b = plant.m_x[V_C_dc];
        double i_c = 0;*/

        double i_a = diode_bridge.m_model.outputs.I_L_a;
        double i_b = diode_bridge.m_model.outputs.I_L_b;
        double i_c = diode_bridge.m_model.outputs.I_L_c;
        fout << timestamp << "," << i_a << "," << i_b << "," << i_c << "\n";
    }
    fout.close();
    std::cout << "Done!\n"
              << std::endl;
}
