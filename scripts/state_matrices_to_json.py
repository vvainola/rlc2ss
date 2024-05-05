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
from state_matrices_to_cpp import StateSpaceMatrices, Diode
import json
import sys
import typing as T

def check_for_invalid_names(component_names: list[str]):
    for name in component_names:
        for name2 in component_names:
            if name in name2 and name != name2:
                sys.exit(f"[ERROR]: Component name \"{name}\" cannot be a substring of \"{name2}\".")


def write_cpp_files(
    model_name: str,
    circuit_combinations: dict[int, StateSpaceMatrices],
    switches: list[str],
    diodes: list[Diode],
    resource_id: int,
    dynamic: bool,
):
    hpp = open(f'{model_name}_matrices.hpp', 'w')
    cpp = open(f'{model_name}_matrices.cpp', 'w')
    ss = circuit_combinations[list(circuit_combinations.keys())[0]]
    check_for_invalid_names(ss.component_names)

    class_name = 'Model_' + os.path.basename(model_name)
    components_list = "\n".join([f'\t\tdouble {str(component)} = -1;' for component in ss.component_names])
    components_compare = " &&\n".join([f'\t\t\t\t{str(component)} == other.{str(component)}' for component in ss.component_names])
    verify_components = "\n".join([f'\t\tassert(components.{str(component)} != -1);' for component in ss.component_names])
    states_list = "\n".join([f'\t\t\tdouble {str(state)};' for state in ss.states])
    inputs_list = "\n".join([f'\t\t\tdouble {str(input)};' for input in ss.inputs])
    outputs_list = "\n".join([f'\t\t\tdouble {str(output)};' for output in ss.outputs])
    switches_list = "\n".join([f'\t\t\tuint64_t {str(switch)} : 1;' for switch in switches])
    update_states = "\n".join([f'\tstates.{state} = outputs.{state};' for state in ss.states])

    template = '''
#pragma once

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include "integrator.h"
#include "nlohmann/json.hpp"
#include <assert.h>

class {class_name} {{
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    union Switches;
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

    union Inputs {{
        Inputs() {{
            data.setZero();
        }}
        struct {{
{inputs_list}
        }};
        Eigen::Vector<double, NUM_INPUTS> data;
    }};

    union Outputs {{
        Outputs() {{
            data.setZero();
        }}
        struct {{
{outputs_list}
        }};
        Eigen::Vector<double, NUM_OUTPUTS> data;
    }};

    union Switches {{
        struct {{
{switches_list}
        }};
        uint64_t all;
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
    Switches switches = {{.all = 0}};

  private:
    void stepInternal(double dt);
    void updateStateSpaceMatrices();

    Integrator<Eigen::Vector<double, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components m_components_DO_NOT_TOUCH;
    Switches m_switches_DO_NOT_TOUCH = {{.all = 0}};
    Eigen::Vector<double, NUM_STATES> m_Bu; // Bu term in "dxdt = Ax + Bu"
    double m_dt_resolution = 0;
    TimestepErrorCorrectionMode m_dt_correction_mode = TimestepErrorCorrectionMode::NONE;
    double m_dt_error_accumulator = 0;
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
    ).replace('\t', '    '))
    hpp.close()

    cpp.write(f'''
#include "{os.path.basename(model_name)}_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <fstream>
#include <format>

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

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

static std::optional<rlc2ss::ZeroCrossingEvent> checkZeroCrossingEvents({class_name}& circuit, {class_name}::Outputs const& prev_outputs) {{
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;
''')

    for diode in diodes:
        # Handle either node being ground
        pos_node = f'circuit.outputs.{diode.pos_node}'
        prev_pos_node = f'prev_outputs.{diode.pos_node}'
        if diode.pos_node == '0':
            pos_node = '0'
            prev_pos_node = '0'
        neg_node = f'circuit.outputs.{diode.neg_node}'
        prev_neg_node = f'prev_outputs.{diode.neg_node}'
        if diode.neg_node == '0':
            neg_node = '0'
            prev_neg_node = '0'

        cpp.write(f'''
    double V_{diode.name} = {pos_node} - {neg_node};
    if (V_{diode.name} > 0 && !circuit.switches.{diode.switch}) {{
        double V_{diode.name}_prev = {prev_pos_node} - {prev_neg_node};
        events.push(rlc2ss::ZeroCrossingEvent{{
            .time = rlc2ss::calcZeroCrossingTime(V_{diode.name}_prev, V_{diode.name}),
            .event_callback = [&]() {{
                circuit.switches.{diode.switch} = 1;
            }}
        }});
    }}
    if (circuit.outputs.{diode.current} < 0 && circuit.switches.{diode.switch}) {{
        events.push(rlc2ss::ZeroCrossingEvent{{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.{diode.current}, circuit.outputs.{diode.current}),
            .event_callback = [&]() {{
                circuit.switches.{diode.switch} = 0;
            }}
        }});
    }}
''')

    cpp.write(f'''
    if (events.size() > 0) {{
        return events.top();
    }}
    return std::nullopt;
}}

{class_name}::{class_name}(Components const& c)
    : components(c),
      m_components_DO_NOT_TOUCH(c) {{
    updateStateSpaceMatrices();
    m_solver.updateJacobian(m_ss.A);
}}
''')

    cpp.write(f'''
void {class_name}::step(double dt, Inputs const& inputs_) {{
    inputs.data = inputs_.data;

    // Copy previous state and outputs if step needs to be redone
    {class_name}::States prev_state;
    {class_name}::Outputs prev_outputs;
    prev_state.data = states.data;
    prev_outputs.data = outputs.data;

    stepInternal(dt);
    std::optional<rlc2ss::ZeroCrossingEvent> zc_event = checkZeroCrossingEvents(*this, prev_outputs);
    while (zc_event) {{
        // Redo step
        states.data = prev_state.data;
        stepInternal(zc_event->time * dt);
        // Process event
        zc_event->event_callback();
        // Run remaining time
        prev_state.data = states.data;
        prev_outputs.data = outputs.data;
        dt = dt * (1 - zc_event->time);
        stepInternal(dt);
        // Check for new events
        zc_event = checkZeroCrossingEvents(*this, prev_outputs);
    }}
}}

void {class_name}::stepInternal(double dt) {{
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != m_components_DO_NOT_TOUCH || switches.all != m_switches_DO_NOT_TOUCH.all) {{
{verify_components}
        m_components_DO_NOT_TOUCH = components;
        m_switches_DO_NOT_TOUCH.all = switches.all;
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
    m_dt_prev = dt;

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
            return t.components == components && t.switches.all == switches.all;
        }});
    if (it != state_space_cache.end()) {{
        m_ss = *it->state_space;
        return;
    }}
''')

    netlist_abspath_without_extension = os.path.abspath(model_name)
    netlist_abspath = f'{netlist_abspath_without_extension}.cir'.replace("\\", "\\\\")
    json_abspath = f'{netlist_abspath_without_extension}_matrices.json'.replace("\\", "\\\\")
    rlc2ss_py = f'{os.path.dirname(os.path.realpath(__file__))}\\rlc2ss.py.'.replace("\\", "\\\\")
    python = sys.executable.replace("\\", "\\\\")
    if dynamic:
        cpp.write(f'''
    //if (m_circuit_json.empty()) {{
    //    m_circuit_json = nlohmann::json::parse(rlc2ss::loadTextResource({resource_id}));
    //}}
    if (!m_circuit_json.contains(std::to_string(switches.all))) {{
        m_circuit_json = nlohmann::json::parse(std::ifstream("{json_abspath}"));
        if (!m_circuit_json.contains(std::to_string(switches.all))) {{
            system(std::format("{python} {rlc2ss_py} {netlist_abspath} --combination={{}}", switches.all).c_str());
        }}
        m_circuit_json = nlohmann::json::parse(std::ifstream("{json_abspath}"));
    }}
    ''')
    else:
        cpp.write(f'''
    if (m_circuit_json.empty()) {{
        m_circuit_json = nlohmann::json::parse(rlc2ss::loadTextResource({resource_id}));
    }}''')

    cpp.write(f'''
    assert(m_circuit_json.contains(std::to_string(switches.all)));

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = m_circuit_json[std::to_string(switches.all)].dump();
''')

    for component in ss.component_names:
        cpp.write(f"\ts = rlc2ss::replace(s, \"{component}\", std::to_string(components.{component}));\n")

    cpp.write(f'''
    // Parse json for the intermediate matrices
    nlohmann::json j = nlohmann::json::parse(s);
    std::string K1_str = j["K1"];
    std::string K2_str = j["K2"];
    std::string A1_str = j["A1"];
    std::string B1_str = j["B1"];
    std::string C1_str = j["C1"];
    std::string D1_str = j["D1"];

    // Create eigen matrices
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES, Eigen::RowMajor> K1(rlc2ss::getCommaDelimitedValues(K1_str).data());
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES, Eigen::RowMajor> K2(rlc2ss::getCommaDelimitedValues(K2_str).data());
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES, Eigen::RowMajor> A1(rlc2ss::getCommaDelimitedValues(A1_str).data());
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_INPUTS, Eigen::RowMajor> B1(rlc2ss::getCommaDelimitedValues(B1_str).data());
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES, Eigen::RowMajor> C1(rlc2ss::getCommaDelimitedValues(C1_str).data());
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_INPUTS, Eigen::RowMajor> D1(rlc2ss::getCommaDelimitedValues(D1_str).data());

    {class_name}_Topology& topology = state_space_cache.emplace_back({class_name}_Topology{{
        .components = components,
        .switches = switches,
        .state_space = calcStateSpace(K1, A1, B1, K2, C1, D1)}});

    m_ss = *topology.state_space;
}}
''')
    cpp.close()


def matrices_to_cpp(
    model_name: str,
    circuit_combinations: dict[int, StateSpaceMatrices],
    switches: list[str],
    diodes: list[Diode],
    resource_id: int | None,
    dynamic: bool,
):
    ss = circuit_combinations[list(circuit_combinations.keys())[0]]
    if resource_id != None:
        write_cpp_files(model_name, circuit_combinations, switches, diodes, resource_id, dynamic)
        circuits = {}
    else:
        circuits = json.load(open(f"{model_name}_matrices.json", "r"))

    write_components = ''
    for component in ss.component_names:
        write_components += f'\tdouble {component} = c.{component};\n'
    for i in sorted(circuit_combinations):
        ss = circuit_combinations[i]
        K1 = str(ss.K1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('*', ' * ')
        K2 = str(ss.K2).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('*', ' * ')
        A1 = str(ss.A1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('*', ' * ')
        B1 = str(ss.B1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('*', ' * ')
        C1 = str(ss.C1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('*', ' * ')
        D1 = str(ss.D1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',').replace('*', ' * ')

        circuits[str(i)] = {}
        circuits[str(i)]["K1"] = K1
        circuits[str(i)]["K2"] = K2
        circuits[str(i)]["A1"] = A1
        circuits[str(i)]["B1"] = B1
        circuits[str(i)]["C1"] = C1
        circuits[str(i)]["D1"] = D1

    with open(f"{model_name}_matrices.json", "w") as outfile:
        json.dump(circuits, outfile, indent=4)

    if resource_id != None:
        with open(f"{model_name}_matrices.rc", "w") as outfile:
            name = os.path.basename(model_name)
            outfile.write(f'#define {name}_matrices_json {resource_id} \n{name}_matrices_json    TEXT    "{name}_matrices.json"')
