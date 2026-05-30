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
#include "qucs\diode_matrices.hpp"
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
Model_saturating_inductor circuit(Model_saturating_inductor::Components{});
#elif defined MUTUAL_INDUCTOR
Model_mutual_inductor circuit(Model_mutual_inductor::Components{
    .Cf = 100e-6,
    .FSRC1 = -100.0,
    .K12 = 0.5,
    .K21 = 0.5,
    .K31 = 0.5,
    .L1 = 1,
    .L2 = 1,
    .L3 = 1,
    .R1 = 10.0,
    .R2 = 10.0,
    .R3 = 10e-3,
    .R4 = 10.0,
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
    .C_a = 10e-3,
    .C_b = 10e-3,
    .C_c = 10e-3,
    .C_n = 100e-3,
    .C_p = 100e-3,
    .L_a = 100e-6,
    .L_b = 100e-6,
    .L_c = 100e-6,
    .L_g_a = 100e-6,
    .L_g_b = 100e-6,
    .L_g_c = 100e-6,
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
    .R_g_a = 1,
    .R_g_b = 1,
    .R_g_c = 1,
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
    // DbgGui_create(dt);
    DbgGui_startUpdateLoop();
#if defined SATURATING_INDUCTOR
    std::vector<double> currents = {0, 1, 2, 5};
    std::vector<double> flux = {0, 0.01, 0.015, 0.0151};
    double L0 = (flux[1] - flux[0]) / (currents[1] - currents[0]);
    double L1 = (flux[2] - flux[1]) / (currents[2] - currents[1]);
    double L2 = (flux[3] - flux[2]) / (currents[3] - currents[2]);
    double L1_act = (L1 * L0) / (L0 - L1);
    double L2_act = (L2 * L1_act) / (L1_act - L2);
    double L1_eff = 1 / (1 / L0 + 1 / L1_act);
    double L2_eff = 1 / (1 / L0 + 1 / L1_act + 1 / L2_act);
    circuit.addInductorSaturation(&circuit.components.L0,
                                  {currents[0], currents[1], currents[2]},
                                  {L0, L1_eff, L2_eff});
#elif defined(CONVERTER)
    std::vector<double> currents = {0, 100,          200,          500};
    std::vector<double> flux =     {0, 100 * 100e-6, 100 * 120e-6, 100 * 125e-6};
    double L0 = (flux[1] - flux[0]) / (currents[1] - currents[0]);
    double L1 = (flux[2] - flux[1]) / (currents[2] - currents[1]);
    double L2 = (flux[3] - flux[2]) / (currents[3] - currents[2]);
    double L1_act = (L1 * L0) / (L0 - L1);
    double L2_act = (L2 * L1_act) / (L1_act - L2);
    double L1_eff = 1 / (1 / L0 + 1 / L1_act);
    double L2_eff = 1 / (1 / L0 + 1 / L1_act + 1 / L2_act);
    circuit.addInductorSaturation(&circuit.components.L_a,
                                  {currents[0], currents[1], currents[2]},
                                  {L0, L1_eff, L2_eff});
    circuit.addInductorSaturation(&circuit.components.L_b,
                                  {currents[0], currents[1], currents[2]},
                                  {L0, L1_eff, L2_eff});
    circuit.addInductorSaturation(&circuit.components.L_c,
                                  {currents[0], currents[1], currents[2]},
                                  {L0, L1_eff, L2_eff});
    /*circuit.addInductorSaturation(&circuit.components.L_g_a,
                                  {currents[0], currents[1], currents[2]},
                                  {L0, L1_eff, L2_eff});
    circuit.addInductorSaturation(&circuit.components.L_g_b,
                                  {currents[0], currents[1], currents[2]},
                                  {L0, L1_eff, L2_eff});
    circuit.addInductorSaturation(&circuit.components.L_g_c,
                                  {currents[0], currents[1], currents[2]},
                                  {L0, L1_eff, L2_eff});*/
#endif
}

extern "C" __declspec(dllexport) void DLL_update(double current_time, double dt) {
#if defined DIODE_TEST
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
