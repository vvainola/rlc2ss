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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "DbgGui/dbg_gui_wrapper.h"
#include "qucs/controlled_sources_matrices.hpp"
#include "qucs/converter_matrices.hpp"
#include "qucs/diode_matrices.hpp"
#include "qucs/mutual_inductor_matrices.hpp"
#include "qucs/saturating_inductor_matrices.hpp"

#ifdef _WIN32
#define RLC2SS_EXPORT __declspec(dllexport)
#else
#define RLC2SS_EXPORT __attribute__((visibility("default")))
#endif

namespace {

enum CircuitModel {
    DIODE = 0,
    SATURATING_INDUCTOR = 1,
    MUTUAL_INDUCTOR = 2,
    CONTROLLED_SOURCES = 3,
    CONVERTER = 4,
};

class Circuit {
  public:
    virtual ~Circuit() = default;
    virtual int inputCount() const = 0;
    virtual int outputCount() const = 0;
    virtual int switchCount() const = 0;
    virtual double* inputs() = 0;
    virtual double* outputs() = 0;
    virtual void init() {}
    virtual void updateSwitches(uint32_t switches) = 0;
    virtual void step(double dt) = 0;
};

template <typename Model>
class CircuitInstance : public Circuit {
  public:
    explicit CircuitInstance(typename Model::Components components)
        : model(components) {}

    int inputCount() const override { return Model::NUM_INPUTS; }
    int outputCount() const override { return Model::NUM_OUTPUTS; }
    int switchCount() const override { return Model::NUM_SWITCHES; }
    double* inputs() override { return model.inputs.data.data(); }
    double* outputs() override { return model.outputs.data.data(); }
    void init() override {}
    void updateSwitches(uint32_t) override {}
    void step(double dt) override { model.step(dt, model.inputs); }

    Model model;
};

template <>
void CircuitInstance<Model_diode>::updateSwitches(uint32_t switches) {
    model.switches.S1 = switches & (1U << 0);
}

template <>
void CircuitInstance<Model_saturating_inductor>::updateSwitches(
    uint32_t switches) {
    model.switches.S1 = switches & (1U << 0);
    model.switches.S2 = switches & (1U << 1);
}

template <>
void CircuitInstance<Model_saturating_inductor>::init() {
    std::vector<double> currents = {0, 1, 2, 5};
    std::vector<double> flux = {0, 0.01, 0.015, 0.0151};
    const double L0 = (flux[1] - flux[0]) / (currents[1] - currents[0]);
    const double L1 = (flux[2] - flux[1]) / (currents[2] - currents[1]);
    const double L2 = (flux[3] - flux[2]) / (currents[3] - currents[2]);
    const double L1_act = (L1 * L0) / (L0 - L1);
    const double L2_act = (L2 * L1_act) / (L1_act - L2);
    const double L1_eff = 1 / (1 / L0 + 1 / L1_act);
    const double L2_eff = 1 / (1 / L0 + 1 / L1_act + 1 / L2_act);
    model.addInductorSaturation(&model.components.L0,
                                {currents[0], currents[1], currents[2]},
                                {L0, L1_eff, L2_eff});
}

template <>
void CircuitInstance<Model_converter>::updateSwitches(uint32_t switches) {
    constexpr double on_delay = 15e-6;
    model.switches.S_D_a_n.setOnOffDelays(on_delay, 0);
    model.switches.S_D_a_p.setOnOffDelays(on_delay, 0);
    model.switches.S_D_b_n.setOnOffDelays(on_delay, 0);
    model.switches.S_D_b_p.setOnOffDelays(on_delay, 0);
    model.switches.S_D_c_n.setOnOffDelays(on_delay, 0);
    model.switches.S_D_c_p.setOnOffDelays(on_delay, 0);

    model.switches.S_D_a_n = switches & (1U << 0);
    model.switches.S_D_a_p = switches & (1U << 1);
    model.switches.S_D_b_n = switches & (1U << 2);
    model.switches.S_D_b_p = switches & (1U << 3);
    model.switches.S_D_c_n = switches & (1U << 4);
    model.switches.S_D_c_p = switches & (1U << 5);
}

template <>
void CircuitInstance<Model_converter>::init() {
    std::vector<double> currents = {0, 100, 200, 500};
    std::vector<double> flux =
        {0, 100 * 100e-6, 100 * 120e-6, 100 * 125e-6};
    const double L0 = (flux[1] - flux[0]) / (currents[1] - currents[0]);
    const double L1 = (flux[2] - flux[1]) / (currents[2] - currents[1]);
    const double L2 = (flux[3] - flux[2]) / (currents[3] - currents[2]);
    const double L1_act = (L1 * L0) / (L0 - L1);
    const double L2_act = (L2 * L1_act) / (L1_act - L2);
    const double L1_eff = 1 / (1 / L0 + 1 / L1_act);
    const double L2_eff = 1 / (1 / L0 + 1 / L1_act + 1 / L2_act);
    const std::vector<double> saturation_currents =
        {currents[0], currents[1], currents[2]};
    const std::vector<double> saturation_inductances =
        {L0, L1_eff, L2_eff};
    model.addInductorSaturation(
        &model.components.L_a, saturation_currents, saturation_inductances);
    model.addInductorSaturation(
        &model.components.L_b, saturation_currents, saturation_inductances);
    model.addInductorSaturation(
        &model.components.L_c, saturation_currents, saturation_inductances);
}

union CircuitStorage {
    char empty;
    CircuitInstance<Model_diode> diode;
    CircuitInstance<Model_saturating_inductor> saturating_inductor;
    CircuitInstance<Model_mutual_inductor> mutual_inductor;
    CircuitInstance<Model_controlled_sources> controlled_sources;
    CircuitInstance<Model_converter> converter;

    CircuitStorage()
        : empty{} {}
    ~CircuitStorage() {}
};

CircuitStorage circuit;
Circuit* active_circuit = nullptr;
std::optional<CircuitModel> active_model;
double debug[20]{};
uint32_t switch_mask = 0;

void destroyCircuit() {
    if (!active_model) {
        return;
    }

    switch (*active_model) {
    case DIODE:
        std::destroy_at(&circuit.diode);
        break;
    case SATURATING_INDUCTOR:
        std::destroy_at(&circuit.saturating_inductor);
        break;
    case MUTUAL_INDUCTOR:
        std::destroy_at(&circuit.mutual_inductor);
        break;
    case CONTROLLED_SOURCES:
        std::destroy_at(&circuit.controlled_sources);
        break;
    case CONVERTER:
        std::destroy_at(&circuit.converter);
        break;
    }

    active_circuit = nullptr;
    active_model.reset();
}

Circuit* constructCircuit(int model) {
    switch (model) {
    case DIODE:
        destroyCircuit();
        active_model = DIODE;
        return std::construct_at(
            &circuit.diode,
            Model_diode::Components{
                .R_D1 = 1e-3,
                .R_D2 = 1e-3,
                .R_D3 = 1e-3,
            });
    case SATURATING_INDUCTOR:
        destroyCircuit();
        active_model = SATURATING_INDUCTOR;
        return std::construct_at(
            &circuit.saturating_inductor,
            Model_saturating_inductor::Components{});
    case MUTUAL_INDUCTOR:
        destroyCircuit();
        active_model = MUTUAL_INDUCTOR;
        return std::construct_at(
            &circuit.mutual_inductor,
            Model_mutual_inductor::Components{
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
    case CONTROLLED_SOURCES:
        destroyCircuit();
        active_model = CONTROLLED_SOURCES;
        return std::construct_at(
            &circuit.controlled_sources,
            Model_controlled_sources::Components{
                .C_1 = 100e-3,
                .C_2 = 10e-3,
                .ESRC3 = 1,
                .FSRC5 = -2.0,
                .GSRC1 = 30.0,
                .HSRC4 = 10.0,
                .L1 = 0.1,
            });
    case CONVERTER:
        destroyCircuit();
        active_model = CONVERTER;
        return std::construct_at(
            &circuit.converter,
            Model_converter::Components{
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
    default:
        return nullptr;
    }
}

} // namespace

extern "C" RLC2SS_EXPORT int DLL_input_count = 0;
extern "C" RLC2SS_EXPORT int DLL_output_count = 0;
extern "C" RLC2SS_EXPORT int DLL_switch_count = 0;
extern "C" RLC2SS_EXPORT double* DLL_inputs = nullptr;
extern "C" RLC2SS_EXPORT uint32_t* DLL_switches = &switch_mask;
extern "C" RLC2SS_EXPORT double* DLL_outputs = nullptr;
extern "C" RLC2SS_EXPORT double* DLL_debug = debug;

extern "C" RLC2SS_EXPORT int DLL_select_model(int model) {
    Circuit* selected = constructCircuit(model);
    if (!selected) {
        return 0;
    }

    active_circuit = selected;
    DLL_input_count = active_circuit->inputCount();
    DLL_output_count = active_circuit->outputCount();
    DLL_switch_count = active_circuit->switchCount();
    DLL_inputs = active_circuit->inputs();
    DLL_outputs = active_circuit->outputs();
    switch_mask = 0;
    for (double& value : debug) {
        value = 0;
    }
    return 1;
}

extern "C" RLC2SS_EXPORT void DLL_init(double dt) {
    if (!active_circuit) {
        return;
    }
    active_circuit->init();
    DbgGui_create(dt);
    DbgGui_startUpdateLoop();
}

extern "C" RLC2SS_EXPORT void DLL_update(double current_time, double dt) {
    if (!active_circuit) {
        return;
    }
    active_circuit->updateSwitches(switch_mask);
    active_circuit->step(dt);
    DbgGui_sampleWithTimestamp(current_time);
}

extern "C" RLC2SS_EXPORT void DLL_terminate() {
    DbgGui_close();
    destroyCircuit();
    DLL_input_count = 0;
    DLL_output_count = 0;
    DLL_switch_count = 0;
    DLL_inputs = nullptr;
    DLL_outputs = nullptr;
}
