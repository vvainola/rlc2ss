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
#include "..\schematics\RL3_matrices.hpp"
#include "..\qucs\diode_matrices.hpp"
#include "qucs\saturating_inductor_matrices.hpp"
#include "DbgGui/dbg_gui_wrapper.h"

// #define DIODE_TEST
// #define RL3
#define SATURATING_INDUCTOR

#if defined RL3
Model_RL3 circuit(
    {.L_a = 1,
     .L_b = 1,
     .L_c = 1,
     .R_a = 10,
     .R_b = 10,
     .R_c = 10,
     .Kab = 0.9934,
     .Kbc = 0.9934,
     .Kca = 0.9934});
#endif
#if defined DIODE_TEST
Model_diode circuit(Model_diode::Components{
    .L1 = 1e-2,
    .R_D1 = 1e-6,
    .R_D2 = 1e-6,
    .R1 = 1,
    .R2 = 1e-3,
    .R3 = 1,
});
#endif
#if defined SATURATING_INDUCTOR
double L0 = 0.01;
double L1 = (0.015 - 0.01) / (2 - 1);
double L2 = (0.0151 - 0.015) / (5 - 2);
double L1_act = (L1 * L0) / (L0 - L1);
double L2_act = (L2 * L1_act) / (L1_act - L2);
Model_saturating_inductor circuit(Model_saturating_inductor::Components{
    .L0 = L0,
    .L1 = L1_act,
    .L2 = L2_act,
    .R = 0.1,
});
#endif

double debug[20];
uint32_t temp;

extern "C" __declspec(dllexport) int DLL_input_count = circuit.NUM_INPUTS;
extern "C" __declspec(dllexport) int DLL_output_count = circuit.NUM_OUTPUTS;
extern "C" __declspec(dllexport) int DLL_switch_count = circuit.NUM_SWITCHES;
extern "C" __declspec(dllexport) double* DLL_inputs = (double*)&circuit.inputs;
extern "C" __declspec(dllexport) uint32_t* DLL_switches = &temp; //(uint32_t*)&circuit.switches2;
extern "C" __declspec(dllexport) double* DLL_outputs = (double*)&circuit.outputs;
extern "C" __declspec(dllexport) double* DLL_debug = debug;

extern "C" __declspec(dllexport) void DLL_init(double dt) {
    //DbgGui_create(dt);
    DbgGui_startUpdateLoop();
}

extern "C" __declspec(dllexport) void DLL_update(double current_time, double dt) {
    double sum = abs(circuit.outputs.I_L0 + circuit.outputs.I_L1 + circuit.outputs.I_L2);
    circuit.switches.S1 = abs(sum) > 1;
    circuit.switches.S2 = abs(sum) > 2;
    circuit.step(dt, circuit.inputs);
    DbgGui_sampleWithTimestamp(current_time);
}

extern "C" __declspec(dllexport) void DLL_terminate() {
    DbgGui_close();
}
