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
#include "qucs\mutual_inductor_matrices.hpp"
#include "qucs\controlled_sources_matrices.hpp"
#include "qucs\converter_matrices.hpp"
#include "DbgGui/dbg_gui_wrapper.h"

// #define DIODE_TEST
// #define RL3
// #define SATURATING_INDUCTOR
// #define MUTUAL_INDUCTOR
// #define CONTROLLED_SOURCES
#define CONVERTER

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
#elif defined DIODE_TEST
Model_diode circuit(Model_diode::Components{
    .L1 = 1e-2,
    .L2 = 1e-2,
    .R1 = 0.1,
    .R2 = 1.0,
    .R3 = 1.0,
    .R_D2 = 1e-3,
    .R_D3 = 1e-3,
    .R_D4 = 1e-3,
});
#elif defined SATURATING_INDUCTOR
double L0 = 0.01;
double L1 = (0.015 - 0.01) / (2 - 1);
double L2 = (0.0151 - 0.015) / (5 - 2);
double L1_act = (L1 * L0) / (L0 - L1);
double L2_act = (L2 * L1_act) / (L1_act - L2);
Model_saturating_inductor circuit(Model_saturating_inductor::Components{});
#elif defined MUTUAL_INDUCTOR
Model_mutual_inductor circuit(Model_mutual_inductor::Components{
    .FSRC1 = -100.0,
    .K12 = 0.5,
    .K21 = 0.5,
    .K31 = 0.5,
});
#elif defined CONTROLLED_SOURCES
Model_controlled_sources circuit(Model_controlled_sources::Components{
    .C_1 = 100e-3,
    .C_2 = 10e-3,
    .ESRC3 = 1,
    .FSRC5 = -2.0,
    .GSRC1 = 30.0,
    .HSRC4 = 10.0,
    .L1 = 0.1,
});
#elif defined CONVERTER
Model_converter circuit(Model_converter::Components{
    .C_n = 100e-3,
    .C_p = 100e-3,
    .L_a = 100e-6,
    .L_b = 100e-6,
    .L_c = 100e-6,
    .R_D_a_n = 1e-3,
    .R_D_a_p = 1e-3,
    .R_D_b_n = 1e-3,
    .R_D_b_p = 1e-3,
    .R_D_c_n = 1e-3,
    .R_D_c_p = 1e-3,
    .R_a = 1,
    .R_b = 1,
    .R_c = 1,
    .R_dc = 1,
    .R_n_p = 1e3,
    .R_n_s = 10e-3,
    .R_p_p = 1e3,
    .R_p_s = 10e-3,

});
#endif

double debug[20];
uint32_t temp;

extern "C" __declspec(dllexport) int DLL_input_count = circuit.NUM_INPUTS;
extern "C" __declspec(dllexport) int DLL_output_count = circuit.NUM_OUTPUTS;
extern "C" __declspec(dllexport) int DLL_switch_count = circuit.NUM_SWITCHES;
extern "C" __declspec(dllexport) double* DLL_inputs = (double*)&circuit.inputs;
extern "C" __declspec(dllexport) uint32_t* DLL_switches = (uint32_t*)&temp; //(uint32_t*)&circuit.switches2;
extern "C" __declspec(dllexport) double* DLL_outputs = (double*)&circuit.outputs;
extern "C" __declspec(dllexport) double* DLL_debug = debug;

extern "C" __declspec(dllexport) void DLL_init(double dt) {
    DbgGui_create(dt);
    DbgGui_startUpdateLoop();
}

extern "C" __declspec(dllexport) void DLL_update(double current_time, double dt) {
#if defined SATURATING_INDUCTOR
    double sum = abs(circuit.outputs.I_L0 + circuit.outputs.I_L1 + circuit.outputs.I_L2);
    circuit.switches.S1 = abs(sum) > 1;
    circuit.switches.S2 = abs(sum) > 2;
#elif defined DIODE_TEST
    circuit.inputs.V_D2 = 0.1;
    circuit.inputs.V_D3 = 0.1;
    circuit.inputs.V_D4 = 0.1;
    circuit.switches.S1 = temp & 1;
#elif defined CONVERTER
    double on_delay = 15e-6;
    circuit.switches.S_a_n.setOnOffDelays(on_delay, 0);
    circuit.switches.S_a_p.setOnOffDelays(on_delay, 0);
    circuit.switches.S_b_n.setOnOffDelays(on_delay, 0);
    circuit.switches.S_b_p.setOnOffDelays(on_delay, 0);
    circuit.switches.S_c_n.setOnOffDelays(on_delay, 0);
    circuit.switches.S_c_p.setOnOffDelays(on_delay, 0);

    circuit.switches.S_a_n = temp & 1 << 6;
    circuit.switches.S_a_p = temp & 1 << 7;
    circuit.switches.S_b_n = temp & 1 << 8;
    circuit.switches.S_b_p = temp & 1 << 9;
    circuit.switches.S_c_n = temp & 1 << 10;
    circuit.switches.S_c_p = temp & 1 << 11;
#endif
    circuit.step(dt, circuit.inputs);
    DbgGui_sampleWithTimestamp(current_time);
}

extern "C" __declspec(dllexport) void DLL_terminate() {
    DbgGui_close();
}
