# MIT License
#
# Copyright (c) 2022 vvainola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
from dataclasses import dataclass
import json
import sys
import typing as T
import sympy

@dataclass
class Diode:
    name: str
    pos_node: str
    neg_node: str
    forward_voltage: str
    current: str
    switch: str

@dataclass
class StateSpaceMatrices:
    component_names: list[str]
    default_values: dict[str, float]
    states: list[sympy.Symbol]
    inputs: list[sympy.Symbol]
    outputs: list[sympy.Symbol]
    K1: sympy.Matrix
    K2: sympy.Matrix
    A1: sympy.Matrix
    B1: sympy.Matrix
    C1: sympy.Matrix
    D1: sympy.Matrix

TAB = "    "

def check_for_invalid_names(component_names: list[str]):
    for name in component_names:
        for name2 in component_names:
            if name in name2 and name != name2:
                sys.exit(f"[ERROR]: Component name \"{name}\" cannot be a substring of \"{name2}\".")


def write_cpp_files(
    netlist: str,
    model_name: str,
    circuit_combinations: dict[int, StateSpaceMatrices],
    switches: list[str],
    diodes: list[Diode],
    dynamic: bool,
):
    hpp = open(f'{model_name}_matrices.hpp', 'w')
    cpp = open(f'{model_name}_matrices.cpp', 'w')
    ss = circuit_combinations[list(circuit_combinations.keys())[0]]
    check_for_invalid_names(ss.component_names)

    model_basename = os.path.basename(model_name)
    class_name = 'Model_' + model_basename
    components_list = "\n".join([f'{TAB*2}double {str(component)} = {ss.default_values.get(str(component), -1)};' for component in ss.component_names])
    components_compare = " &&\n".join([f'{TAB*4}{str(component)} == other.{str(component)}' for component in ss.component_names])
    verify_components = "\n".join([f'{TAB*2}assert(components.{str(component)} != -1);' for component in ss.component_names])
    states_list = "\n".join([f'{TAB*3}double {str(state)};' for state in ss.states])
    inputs_list = "\n".join([f'{TAB*3}double {str(input)};' for input in ss.inputs])
    outputs_list = "\n".join([f'{TAB*3}double {str(output)};' for output in ss.outputs])
    switches_list = "\n".join([f'{TAB*2}rlc2ss::OnOffDelay {str(switch)};' for switch in switches])
    update_states = "\n".join([f'{TAB}states.{state} = outputs.{state};' for state in ss.states])
    if len(switches) > 0:
        switches_to_int = "0 |" + " |".join(f"\n{TAB*2}({switch} << {i})" for i, switch in enumerate(switches))
    else:
        switches_to_int = "0"
    switches_min_delay = (
        "std::min({double(rlc2ss::OnOffDelay::MAX_DELAY),\n" + TAB * 5
        + f",\n{TAB * 5}".join(
            f"{switch}.pendingTime()" for switch in switches
        )
        + "})"
    )
    switches_step = f"\n{TAB}".join([f'{str(switch)}.step(dt);' for switch in switches])

    template = '''
#pragma once

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

#include "on_off_delay.hpp"
#include "integrator.hpp"
#include "rlc2ss.h"

#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/LU"

#include "nlohmann/json.hpp"

#include <assert.h>

class {class_name} {{
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    struct Switches;
    struct StateSpaceMatrices;

    {class_name}() {{}}
    {class_name}(Components const& c);

    static inline constexpr size_t NUM_INPUTS = {num_inputs};
    static inline constexpr size_t NUM_OUTPUTS = {num_outputs};
    static inline constexpr size_t NUM_STATES = {num_states};
    static inline constexpr size_t NUM_SWITCHES = {num_switches};

    enum class TimestepErrorCorrectionMode {{
        // Ignore error in timestep length that is not a multiple of timestep resolution. Use this if
        // e.g. resolution is 0.1e-6 and the variation in timestep length is a multiple of that and
        // there should not ever be any error although floating point roundings may cause some.
        NONE,
        // Round the used timestep to closest multiple of resolution and store the error to accumulator
        // so that the timestep length error will be corrected when accumulator becomes a multiple of the
        // timestep resolution.
        ACCUMULATE,
        // The timestep length that is not a multiple of timestep resolution will be integrated with
        // adaptive step size runge-kutta-fehlberg. E.g. If resolution 1us and timestep is 12.1us,
        // 12 us will be solved with Tustin and remaining 0.1us with RKF to avoid calculating jacobian
        // inverse for very small timesteps
        INTEGRATE_ADAPTIVE
    }};

    void setTimestepResolution(double dt, TimestepErrorCorrectionMode mode) {{
        m_solver.enableInverseMatrixCaching(true);
        m_dt_resolution = dt;
        m_dt_correction_mode = mode;
    }}

    void step(double dt, Inputs const& inputs_);

    /// @brief Add stepwise saturation curve to inductor. The inductance is reduced when the current
    /// exceeds the breakpoints and increased when current goes below the breakpoints.
    /// @param inductor Pointer to inductor in component struct e.g. &circuit.components.L0
    /// @param current Current breakpoints in ascending order. First breakpoint must be 0.
    /// @param inductance Inductance values at the breakpoints.
    /// Example:
    /// currents   = {{0,       100,   200,   300}}
    /// inductances = {{100e-6, 75e-6, 50e-6, 25e-6}}
    void addInductorSaturation(double* inductor, std::vector<double> current, std::vector<double> inductance);

    union Inputs {{
        Inputs() {{ data.setZero(); }}
        Inputs(const Inputs& other) {{ data = other.data; }}
        struct {{
{inputs_list}
        }};
        Eigen::Vector<double, NUM_INPUTS> data;
    }};

    union Outputs {{
        Outputs() {{ data.setZero(); }}
        Outputs(const Outputs& other) {{ data = other.data; }}
        struct {{
{outputs_list}
        }};
        Eigen::Vector<double, NUM_OUTPUTS> data;
    }};

    struct Switches {{
{switches_list}

        uint64_t all() const;
        double smallestDelay();
        void step(double dt);
    }};

    struct Components {{
{components_list}

        bool operator==(Components const& other) const {{
            return
{components_compare};
        }}

        bool operator!=(Components const& other) const {{
            return !(*this == other);
        }}
    }};

    union States {{
        States() {{
            data.setZero();
        }}
        States(const States& other) {{
            data = other.data;
        }}
        struct {{
{states_list}
        }};
        Eigen::Vector<double, NUM_STATES> data;
    }};

    struct StateSpaceMatrices {{
        Eigen::Matrix<double, NUM_STATES, NUM_STATES> A;
        Eigen::Matrix<double, NUM_STATES, NUM_INPUTS> B;
        Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES> C;
        Eigen::Matrix<double, NUM_OUTPUTS, NUM_INPUTS> D;
    }};

    Eigen::Vector<double, NUM_STATES> dxdt(Eigen::Vector<double, NUM_STATES> const& state, double /*t*/) const {{
        return m_ss.A * state + m_Bu;
    }}

    Components components;
    Inputs inputs;
    States states;
    Outputs outputs;
    Switches switches;

  private:
    std::optional<rlc2ss::ZeroCrossingEvent> checkZeroCrossingEvents(Outputs const& prev_outputs);
    void stepWithZeroCrossingDetection(double dt);
    void stepModel(double dt);
    void updateStateSpaceMatrices();

    Integrator<Eigen::Vector<double, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components _M_components_DO_NOT_TOUCH;
    Switches _M_switches_DO_NOT_TOUCH;
    Eigen::Vector<double, NUM_STATES> m_Bu; // Bu term in "dxdt = Ax + Bu"
    double m_dt_resolution = 0;
    TimestepErrorCorrectionMode m_dt_correction_mode = TimestepErrorCorrectionMode::NONE;
    double m_dt_error_accumulator = 0;
    using ZeroCrossCallback = std::function<std::optional<rlc2ss::ZeroCrossingEvent>(Outputs const& prev_outputs, Outputs const& new_outputs)>;
    std::vector<ZeroCrossCallback> m_zero_crossing_callbacks;
    // The json file with symbolic intermediate matrices
    nlohmann::json m_circuit_json;

    static_assert(sizeof(double) * NUM_STATES == sizeof(States));
    static_assert(sizeof(double) * NUM_INPUTS == sizeof(Inputs));
    static_assert(sizeof(double) * NUM_OUTPUTS == sizeof(Outputs));
}};

#pragma warning(default : 4127) // conditional expression is constant
#pragma warning(default : 4189) // local variable is initialized but not referenced
#pragma warning(default : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(default : 4408) // anonymous struct did not declare any data members
#pragma warning(default : 5054) // operator '&': deprecated between enumerations of different types
'''
    hpp.write(template.format(
        class_name = class_name,
        num_inputs = len(ss.inputs),
        num_outputs = len(ss.outputs),
        num_states = len(ss.states),
        num_switches = len(switches),
        components_list = components_list,
        components_compare = components_compare,
        verify_components = verify_components,
        states_list = states_list,
        inputs_list = inputs_list,
        outputs_list = outputs_list,
        switches_list = switches_list,
        update_states = update_states,
    ).replace('\t', TAB))
    hpp.close()

    include_json_header = f'#include "{model_basename}_matrices_json.h"' if not dynamic else ''
    cpp.write(f'''
#include "{model_basename}_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <fstream>
#include <format>
#include <memory>
{include_json_header}

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

inline constexpr int MAX_ZERO_CROSS_EVENTS = 100;

static std::unique_ptr<{class_name}::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES> const& K1,
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES> const& A1,
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_INPUTS> const& B1,
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES> const& K2,
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES> const& C1,
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_INPUTS> const& D1) {{
    auto ss = std::make_unique<{class_name}::StateSpaceMatrices>();
    ss->A = K1.partialPivLu().solve(A1);
    ss->B = K1.partialPivLu().solve(B1);
    ss->C = (C1 + K2 * ss->A);
    ss->D = (D1 + K2 * ss->B);
    return ss;
}}

std::optional<rlc2ss::ZeroCrossingEvent> {class_name}::checkZeroCrossingEvents({class_name}::Outputs const& prev_outputs) {{
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;
''')

    # Sort diodes by their name for deterministic ordering
    diodes.sort(key=lambda d: d.name)
    for diode in diodes:
        # Handle either node being ground
        pos_node = f'outputs.{diode.pos_node}'
        prev_pos_node = f'prev_outputs.{diode.pos_node}'
        if diode.pos_node == '0':
            pos_node = '0'
            prev_pos_node = '0'
        neg_node = f'outputs.{diode.neg_node}'
        prev_neg_node = f'prev_outputs.{diode.neg_node}'
        if diode.neg_node == '0':
            neg_node = '0'
            prev_neg_node = '0'

        cpp.write(f'''
    // Diode {diode.name}
    double V_{diode.name} = {pos_node} - {neg_node};
    if (V_{diode.name} > inputs.{diode.forward_voltage} && !switches.{diode.switch}) {{
        double V_{diode.name}_prev = {prev_pos_node} - {prev_neg_node};
        events.push(rlc2ss::ZeroCrossingEvent{{
            .time = rlc2ss::calcZeroCrossingTime(V_{diode.name}_prev, V_{diode.name}),
            .event_callback = [&]() {{
                switches.{diode.switch}.forceOutput(true);
            }}
        }});
    }}
    if (outputs.{diode.current} < 0 && switches.{diode.switch}.outputForced()) {{
        events.push(rlc2ss::ZeroCrossingEvent{{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.{diode.current}, outputs.{diode.current}),
            .event_callback = [&]() {{
                switches.{diode.switch}.forceOutput(std::nullopt);
            }}
        }});
    }}
''')

    cpp.write(f'''
    for (auto const& callback : m_zero_crossing_callbacks) {{
        std::optional<rlc2ss::ZeroCrossingEvent> event = callback(prev_outputs, outputs);
        if (event) {{
            events.push(*event);
        }}
    }}

    if (events.size() > 0) {{
        return events.top();
    }}
    return std::nullopt;
}}

{class_name}::{class_name}(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {{
}}
''')

    cpp.write(f'''
void {class_name}::addInductorSaturation(double* inductor, std::vector<double> currents, std::vector<double> inductances) {{
    // Check that the currents are ascending and inductances are descending
    assert(currents.size() == inductances.size());
    for (int i = 1; i < currents.size(); ++i) {{
        assert(currents[i] >= currents[i - 1]);
        assert(inductances[i] <= inductances[i - 1]);
    }}
    int i_L_output_idx = -1;''')

    for component in ss.component_names:
         if component.startswith('L'):
            cpp.write(f'''
    if (inductor == &components.{component}) {{
        i_L_output_idx = {ss.outputs.index(sympy.Symbol(f'I_{component}'))};
    }}''')
    cpp.write(f'''
    if (i_L_output_idx == -1) {{
        assert(("Invalid pointer to inductor", false));
    }}

    for (int i = 1; i < currents.size(); ++i) {{
        double threshold = currents[i];
        double inductance_prev = inductances[i - 1];
        double inductance = inductances[i];
        // Increase inductance when current goes below level
        m_zero_crossing_callbacks.push_back([=](Outputs const& outputs_prev, Outputs const& outputs_new) -> std::optional<rlc2ss::ZeroCrossingEvent> {{
            double i_prev = fabs(outputs_prev.data[i_L_output_idx]);
            double i_new = fabs(outputs_new.data[i_L_output_idx]);
            if (i_prev > threshold && i_new < threshold) {{
                return rlc2ss::ZeroCrossingEvent{{
                    .time = rlc2ss::calcZeroCrossingTime(i_prev - threshold, i_new - threshold),
                    .event_callback = [&]() {{
                        *inductor = inductance_prev;
                    }}}};
            }}
            return std::nullopt;
        }});
        // Decrease inductance when current goes above level
        m_zero_crossing_callbacks.push_back([=](Outputs const& outputs_prev, Outputs const& outputs_new) -> std::optional<rlc2ss::ZeroCrossingEvent> {{
            double i_prev = fabs(outputs_prev.data[i_L_output_idx]);
            double i_new = fabs(outputs_new.data[i_L_output_idx]);
            if (i_prev < threshold && i_new > threshold) {{
                return rlc2ss::ZeroCrossingEvent{{
                    .time = rlc2ss::calcZeroCrossingTime(i_prev - threshold, i_new - threshold),
                    .event_callback = [&]() {{
                        *inductor = inductance;
                    }}}};
            }}
            return std::nullopt;
        }});
    }}
}}
''')

    cpp.write(f'''
void {class_name}::step(double dt, Inputs const& inputs_) {{
    inputs.data = inputs_.data;

    // Step to the next switching event
    double smallest_dt = switches.smallestDelay();
    while (smallest_dt < dt) {{
        switches.step(smallest_dt);
        stepWithZeroCrossingDetection(smallest_dt);
        dt -= smallest_dt;
        smallest_dt = switches.smallestDelay();
    }}

    // Step remaining time
    switches.step(dt);
    stepWithZeroCrossingDetection(dt);
}}

void {class_name}::stepWithZeroCrossingDetection(double dt) {{
    // No need to do anything
    if (dt < rlc2ss::MINIMUM_TIMESTEP) {{
        return;
    }}

    // Copy previous state and outputs if step needs to be redone
    {class_name}::States prev_state;
    {class_name}::Outputs prev_outputs;
    prev_state.data = states.data;
    prev_outputs.data = outputs.data;

    stepModel(dt);
    std::optional<rlc2ss::ZeroCrossingEvent> zc_event = checkZeroCrossingEvents(prev_outputs);
    int zc_event_count = 0;
    while (zc_event && zc_event_count < MAX_ZERO_CROSS_EVENTS) {{
        zc_event_count++;
        // Redo step
        states.data = prev_state.data;
        stepModel(zc_event->time * dt);
        // Process event
        zc_event->event_callback();
        // Run remaining time
        prev_state.data = states.data;
        prev_outputs.data = outputs.data;
        dt = dt * (1 - zc_event->time);
        stepModel(dt);
        // Check for new events
        zc_event = checkZeroCrossingEvents(prev_outputs);
    }}
}}

void {class_name}::stepModel(double dt) {{
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all() != _M_switches_DO_NOT_TOUCH.all() || !m_solver.initialized()) {{
{verify_components}
        _M_components_DO_NOT_TOUCH = components;
        _M_switches_DO_NOT_TOUCH = switches;
        updateStateSpaceMatrices();
        m_solver.updateJacobian(m_ss.A);
        // Solve one step with backward euler to reduce numerical oscillations
        m_Bu = m_ss.B * inputs.data;
        if (m_dt_resolution > 0) {{
            double multiple = std::round(dt / m_dt_resolution);
            states.data = m_solver.stepBackwardEuler(*this, states.data, 0.0, multiple * m_dt_resolution);
        }} else {{
            states.data = m_solver.stepBackwardEuler(*this, states.data, 0.0, dt);
        }}
    }} else {{
        m_Bu = m_ss.B * inputs.data;

        if (m_dt_resolution > 0) {{
            if (m_dt_correction_mode == TimestepErrorCorrectionMode::NONE) {{
                // Solve with tustin as multiples of resolution and ignore any error
                double multiple = std::round(dt / m_dt_resolution);
                states.data = m_solver.stepTustin(*this, states.data, 0.0, multiple * m_dt_resolution);
            }} else if (m_dt_correction_mode == TimestepErrorCorrectionMode::ACCUMULATE) {{
                // Solve with tustin as multiples of resolution and accumulate error to correct the timestep length
                // on later steps
                double multiple = (dt + m_dt_error_accumulator) / m_dt_resolution;
                m_dt_error_accumulator += dt - std::round(multiple) * m_dt_resolution;
                states.data = m_solver.stepTustin(*this, states.data, 0.0, std::round(multiple) * m_dt_resolution);
            }} else if (m_dt_correction_mode == TimestepErrorCorrectionMode::INTEGRATE_ADAPTIVE) {{
                // Solve with tustin as multiples of resolution and the remaining time with runge-kutta so
                // that the matrix inverses required for implicit integration can be cached for common timesteps
                // and weird small remainders are solved with adaptive integration.
                double multiple = dt / m_dt_resolution;
                if (std::abs(std::round(multiple) - multiple) > 1e-6) {{
                    double dt1 = std::floor(multiple) * m_dt_resolution;
                    double dt2 = (multiple - std::floor(multiple)) * m_dt_resolution;
                    states.data = m_solver.stepTustin(*this, states.data, 0.0, dt1);
                    states.data = m_solver.stepRungeKuttaFehlberg(*this, states.data, 0.0, dt2);
                }} else {{
                    states.data = m_solver.stepTustin(*this, states.data, 0.0, multiple * m_dt_resolution);
                }}
            }}
        }} else {{
            states.data = m_solver.stepTustin(*this, states.data, 0.0, dt);
        }}
    }}

    // Update output
    outputs.data = m_ss.C * states.data + m_ss.D * inputs.data;

    // Update states from outputs to have correct values for dependent states
{update_states}
}}
''')

    cpp.write(f'''
struct {class_name}_Topology {{
    {class_name}::Components components;
    {class_name}::Switches switches;
    std::unique_ptr<{class_name}::StateSpaceMatrices> state_space;
}};

void {class_name}::updateStateSpaceMatrices() {{
    static std::vector<{class_name}_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&]({class_name}_Topology const& t) {{
            return t.components == components && t.switches.all() == switches.all();
        }});
    if (it != state_space_cache.end()) {{
        m_ss = *it->state_space;
        return;
    }}
''')

    if dynamic:
        cpp.write(f'    std::string netlist = \"{netlist}\";\n')
        cpp.write('    std::unordered_map<std::string, double> component_values;\n')
        for component in ss.component_names:
            cpp.write(f'\tcomponent_values["{component}"] = components.{component};\n')

        cpp.write(f'    rlc2ss::StateSpaceMatrices ss = rlc2ss::formStateSpaceMatrices(netlist, int(switches.all()), component_values);\n')
    else:
        cpp.write((f'''
    if (m_circuit_json.empty()) {{
        m_circuit_json = nlohmann::json::parse(std::string({model_basename}_matrices_json_hexdump, {model_basename}_matrices_json_hexdump + {model_basename}_matrices_json_hexdump_len));
    }}
    assert(m_circuit_json.contains(std::to_string(switches.all())));

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = m_circuit_json[std::to_string(switches.all())].dump();\n'''))

        for component in ss.component_names:
            cpp.write(f"{TAB}s = rlc2ss::replace(s, \"{component}\", std::format(\"({{}})\", components.{component}));\n")
        cpp.write((f'''
    // Parse json for the intermediate matrices
    nlohmann::json j = nlohmann::json::parse(s);
    rlc2ss::StateSpaceMatrices ss = {{
        .K1 = j["K1"],
        .K2 = j["K2"],
        .A1 = j["A1"],
        .B1 = j["B1"],
        .C1 = j["C1"],
        .D1 = j["D1"],
    }};'''))

    # The row major template parameter can only be specified if there is more than 1 column
    states_row_major = f", Eigen::RowMajor" if len(ss.states) > 1 else ""
    inputs_row_major = f", Eigen::RowMajor" if len(ss.inputs) > 1 else ""
    cpp.write(f'''
    // Create eigen matrices
    Eigen::Matrix<double, NUM_STATES, NUM_STATES{states_row_major}> K1(rlc2ss::getCommaDelimitedValues(ss.K1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES{states_row_major}> K2(rlc2ss::getCommaDelimitedValues(ss.K2).data());
    Eigen::Matrix<double, NUM_STATES, NUM_STATES{states_row_major}> A1(rlc2ss::getCommaDelimitedValues(ss.A1).data());
    Eigen::Matrix<double, NUM_STATES, NUM_INPUTS{inputs_row_major}> B1(rlc2ss::getCommaDelimitedValues(ss.B1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES{states_row_major}> C1(rlc2ss::getCommaDelimitedValues(ss.C1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_INPUTS{inputs_row_major}> D1(rlc2ss::getCommaDelimitedValues(ss.D1).data());

    {class_name}_Topology& topology = state_space_cache.emplace_back({class_name}_Topology{{
        .components = components,
        .switches = switches,
        .state_space = calcStateSpace(K1, A1, B1, K2, C1, D1)}});

    m_ss = *topology.state_space;
}}

uint64_t {class_name}::Switches::all() const {{
    return {switches_to_int};
}}

double {class_name}::Switches::smallestDelay() {{
    return {switches_min_delay};
}}

void {class_name}::Switches::step(double dt) {{
    {switches_step}
}}
''')
    cpp.close()


def matrices_to_cpp(
    netlist: str,
    model_name: str,
    circuit_combinations: dict[int, StateSpaceMatrices],
    switches: list[str],
    diodes: list[Diode],
    dynamic: bool,
    update_existing: bool,
):
    ss = circuit_combinations[list(circuit_combinations.keys())[0]]
    if not(update_existing):
        write_cpp_files(netlist, model_name, circuit_combinations, switches, diodes, dynamic)
        circuits = {}
    else:
        circuits = json.load(open(f"{model_name}_matrices.json", "r"))
    if dynamic:
        return

    write_components = ''
    for component in ss.component_names:
        write_components += f'\tdouble {component} = c.{component};\n'
    for i in sorted(circuit_combinations):
        ss = circuit_combinations[i]
        K1 = str(ss.K1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('**', '^').replace('*', ' * ')
        K2 = str(ss.K2).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('**', '^').replace('*', ' * ')
        A1 = str(ss.A1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('**', '^').replace('*', ' * ')
        B1 = str(ss.B1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('**', '^').replace('*', ' * ')
        C1 = str(ss.C1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('**', '^').replace('*', ' * ')
        D1 = str(ss.D1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('**', '^').replace('*', ' * ')

        circuits[str(i)] = {}
        circuits[str(i)]["K1"] = K1
        circuits[str(i)]["K2"] = K2
        circuits[str(i)]["A1"] = A1
        circuits[str(i)]["B1"] = B1
        circuits[str(i)]["C1"] = C1
        circuits[str(i)]["D1"] = D1

    with open(f"{model_name}_matrices.json", "w") as outfile:
        json.dump(circuits, outfile, indent=4)
