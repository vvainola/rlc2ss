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
#include "../schematics/LCL_with_faults_S_matrices.h"
#include <bitset>

struct abc {
    double a;
    double b;
    double c;
};

constexpr double PI = 3.141592652;

class Plant {
  public:
    Plant()
        : m_model(Model_LCL_with_faults_S::Components{
            .Lc_a = 0.1,
            .Lc_b = 0.1,
            .Lc_c = 0.1,
            .Lg_a = 1,
            .Lg_b = 1,
            .Lg_c = 1,
            .L_fault_a = 1,
            .L_fault_b = 1,
            .L_fault_c = 1,
            .Ls_a = 1,
            .Ls_b = 1,
            .Ls_c = 1,
            .C_a = 1e-3,
            .C_b = 1e-3,
            .C_c = 1e-3,
            .Rc_a = 1,
            .Rc_b = 1,
            .Rc_c = 1,
            .Rf_a = 0.1,
            .Rf_b = 0.1,
            .Rf_c = 0.1,
            .Rg_a = 1,
            .Rg_b = 1,
            .Rg_c = 1,
            .R_mcb_a = 1,
            .R_mcb_b = 1,
            .R_mcb_c = 1,
            .R_fault_a = 1,
            .R_fault_b = 1,
            .R_fault_c = 1,
            .Rs_a = 1,
            .Rs_b = 1,
            .Rs_c = 1,
        }) {
    }

    void step(double dt, abc uconv, abc ugrid) {
        Model_LCL_with_faults_S::Inputs inputs;
        inputs.V_a = uconv.a;
        inputs.V_b = uconv.b;
        inputs.V_c = uconv.c;
        inputs.Vs_a = ugrid.a;
        inputs.Vs_b = ugrid.b;
        inputs.Vs_c = ugrid.c;
        m_model.step(dt, inputs);
    }

    Model_LCL_with_faults_S m_model;
};

int main() {
    Plant plant;

    double amplitude = 400;

    double freq = 2 * PI * 50;
    double b_offset = -2.0 * PI / 3.0;
    double c_offset = -4.0 * PI / 3.0;
    double angle = PI;
    plant.m_model.switches.S_fault_a = 1;
    plant.m_model.switches.S_fault_b = 1;
    plant.m_model.switches.S_fault_c = 1;
    plant.m_model.switches.S_fault_neutral_to_ground = 1;

    abc u_conv;
    abc u_grid;
    double t_step = 10e-6;
    std::ofstream fout("temp.csv");
    fout << "time,a,b,c\n";
    double t = 0;
    for (; t < 0.1; t += t_step) {
        u_conv.a = amplitude * sin(freq * t);
        u_conv.b = amplitude * sin(freq * t + b_offset);
        u_conv.c = amplitude * sin(freq * t + c_offset);
        u_grid.a = amplitude * sin(freq * t + angle);
        u_grid.b = amplitude * sin(freq * t + angle + b_offset);
        u_grid.c = amplitude * sin(freq * t + angle + c_offset);

        plant.step(t_step, u_conv, u_grid);

        double i_a = plant.m_model.outputs.V3_a;
        double i_b = plant.m_model.outputs.V3_b;
        double i_c = plant.m_model.outputs.V3_c;
        fout << t << "," << i_a << "," << i_b << "," << i_c << "\n";
    }
    fout.close();
    std::cout << "Done!\n"
              << std::endl;
    // system("python ..\\scripts\\plot.py");
}
