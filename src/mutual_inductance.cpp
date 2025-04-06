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

#include "integrator.hpp"

#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include "..\schematics\RL3_matrices.hpp"

struct abc {
    double a;
    double b;
    double c;
};

constexpr double PI = 3.141592652;

int main() {
    Model_RL3 plant(Model_RL3::Components{
        .L_a = 1,
        .L_b = 1,
        .L_c = 1,
        .R_a = 10,
        .R_b = 10,
        .R_c = 10,
        .Kab = 0.5,
        .Kbc = 0.5,
        .Kca = 0.5
        });

    double amplitude = 400;

    double freq = 2 * PI * 50 /*/ w_base*/;
    double b_offset = -2.0 * PI / 3.0;
    double c_offset = -4.0 * PI / 3.0;
    double angle = PI;

    abc u_grid;
    double t_step = 10e-6;
    std::ofstream fout("temp.csv");
    double t = 0;
    fout << "time,a,b,c\n";
    for (; t < 0.2; t += t_step) {
        u_grid.a = amplitude * sin(freq * t + angle);
        u_grid.b = amplitude * sin(freq * t + angle + b_offset);
        u_grid.c = amplitude * sin(freq * t + angle + c_offset);
        Model_RL3::Inputs inputs;
        inputs.V_a = u_grid.a;
        inputs.V_b = u_grid.b;
        inputs.V_c = u_grid.c;

        plant.step(t_step, inputs);

        double i_a = plant.outputs.I_L_a;
        double i_b = plant.outputs.I_L_b;
        double i_c = plant.outputs.I_L_c;
        fout << t << "," << i_a << "," << i_b << "," << i_c << "\n";
    }
    fout.close();
    std::cout << "Done!\n"
              << std::endl;
}
