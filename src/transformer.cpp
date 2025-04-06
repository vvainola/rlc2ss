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
#include "integrator.hpp"

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

    Transformer() : 
        m_model(Model_transformer::Components{
            .L1 = 1e-6,
            .R1 = 1e-6,
            .R2 = 5,
            .E1 = 10,
            .F1 = 1.0 / 10
        })
    {
    }

    void step(double dt, double Vs)
    {
        Model_transformer::Inputs inputs;
        inputs.Vs = Vs;
        m_model.step(dt, inputs);
    }

    Model_transformer m_model;
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

        double i_a = plant.m_model.outputs.I_L1;
        double i_b = plant.m_model.outputs.I_R2;
        double i_c = plant.m_model.outputs.V_R2;
        fout << t << "," << i_a << "," << i_b << "," << i_c << "\n";
    }
    fout.close();
    std::cout << "Done!\n"
              << std::endl;
}
