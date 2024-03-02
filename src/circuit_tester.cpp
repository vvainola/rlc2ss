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

#include "integrator.h"

#include <complex>
#include "..\schematics\RL3_matrices.hpp"
#include "DbgGui/dbg_gui_wrapper.h"

Model_RL3 circuit(
    {.L_a = 1,
     .L_b = 1,
     .L_c = 1,
     .R_a = 10,
     .R_b = 10,
     .R_c = 10,
     .Kab = 0.5,
     .Kbc = 0.5,
     .Kca = 0.5});
double debug[20];

extern "C" __declspec(dllexport) int DLL_input_count = circuit.NUM_INPUTS;
extern "C" __declspec(dllexport) int DLL_output_count = circuit.NUM_OUTPUTS;
extern "C" __declspec(dllexport) int DLL_switch_count = circuit.NUM_SWITCHES;
extern "C" __declspec(dllexport) double* DLL_inputs = (double*)&circuit.inputs;
extern "C" __declspec(dllexport) double* DLL_switches = (double*)&circuit.switches;
extern "C" __declspec(dllexport) double* DLL_outputs = (double*)&circuit.outputs;
extern "C" __declspec(dllexport) double* DLL_debug = debug;

extern "C" __declspec(dllexport) void DLL_init(double dt) {
    DbgGui_create(dt);
    DbgGui_startUpdateLoop();
}

extern "C" __declspec(dllexport) void DLL_update(double current_time, double dt) {
    DbgGui_sampleWithTimestamp(current_time);
    circuit.step(dt, circuit.inputs);
}

extern "C" __declspec(dllexport) void DLL_terminate() {
    DbgGui_close();
}
