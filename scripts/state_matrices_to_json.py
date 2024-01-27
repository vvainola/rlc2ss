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
from state_matrices_to_cpp import StateSpaceMatrices
import json
import sys

def check_for_invalid_names(component_names: list[str]):
    for name in component_names:
        for name2 in component_names:
            if name in name2 and name != name2:
                sys.exit(f"[ERROR]: Component name \"{name}\" cannot be a substring of \"{name2}\".")

def matrices_to_cpp(model_name: str, circuit_combinations: dict[int, StateSpaceMatrices], switches: list[str], resource_id: int):
    hpp = open(f'{model_name}_matrices.hpp', 'w')
    cpp = open(f'{model_name}_matrices.cpp', 'w')
    ss = circuit_combinations[0]
    check_for_invalid_names(ss.component_names)

    class_name = 'Model_' + os.path.basename(model_name)
    components_list = "\n".join([f'\t\tdouble {str(component)} = -1;' for component in ss.component_names])
    components_compare = " &&\n".join([f'\t\t\t\t{str(component)} == other.{str(component)}' for component in ss.component_names])
    verify_components = "\n".join([f'\t\t\tassert(components.{str(component)} != -1);' for component in ss.component_names])
    states_list = "\n".join([f'\t\t\tdouble {str(state)};' for state in ss.states])
    inputs_list = "\n".join([f'\t\t\tdouble {str(input)};' for input in ss.inputs])
    outputs_list = "\n".join([f'\t\t\tdouble {str(output)};' for output in ss.outputs])
    switches_list = "\n".join([f'\t\t\tuint32_t {str(switch)} : 1;' for switch in switches])
    update_states = "\n".join([f'\t\tstates.{state} = outputs.{state};' for state in ss.states])

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
#include <assert.h>

class {class_name} {{
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    union Switches;
    struct StateSpaceMatrices;

    {class_name}(){{}}
    {class_name}(Components const& c);

    static inline constexpr size_t NUM_INPUTS = {num_inputs};
    static inline constexpr size_t NUM_OUTPUTS = {num_outputs};
    static inline constexpr size_t NUM_STATES = {num_states};
    static inline constexpr size_t NUM_SWITCHES = {num_switches};

    Eigen::Vector<double, NUM_STATES> dxdt(Eigen::Vector<double, NUM_STATES> const& state, double /*t*/) const {{
        return m_ss.A * state + m_Bu;
    }}

    Eigen::Matrix<double, NUM_STATES, NUM_STATES> const& jacobian(const Eigen::Vector<double, NUM_STATES>& /*state*/, const double& /*t*/) const {{
        return m_ss.A;
    }}

    // Implicit Tustin integration is used only for this timestep and different length timesteps are solved with adaptive step size integration.
    void setImplicitIntegrationTimestep(double dt) {{
        if (dt != m_dt_implicit && dt > 0) {{
            m_solver.update_tustin_coeffs(jacobian(states.data, dt), dt);
        }}
        m_dt_implicit = dt;
    }}

    void step(double dt, Inputs const& inputs_) {{
        inputs.data = inputs_.data;
        // Update state-space matrices if needed
        if (components != m_components_DO_NOT_TOUCH || switches.all != m_switches_DO_NOT_TOUCH.all) {{
{verify_components}
            m_components_DO_NOT_TOUCH = components;
            m_switches_DO_NOT_TOUCH.all = switches.all;
            m_ss = calculateStateSpace(components, switches);
            // Solve one step with backward euler to reduce numerical oscillations
            m_Bu = m_ss.B * inputs.data;
            states.data = m_solver.step_backward_euler(*this, states.data, 0.0, dt);

            // Update coefficients to make following steps with Tustin
            if (m_dt_implicit > 0) {{
                m_solver.update_tustin_coeffs(jacobian(states.data, m_dt_implicit), m_dt_implicit);
            }} else {{
                m_solver.update_tustin_coeffs(jacobian(states.data, dt), dt);
            }}
        }} else {{
            m_Bu = m_ss.B * inputs.data;
            // Coefficient need to be updated if dt changes and implicit integration is used for all step sizes
            if (dt != m_dt_prev && m_dt_implicit == 0) {{
                m_solver.update_tustin_coeffs(jacobian(states.data, dt), dt);
            }}

            if (dt != m_dt_implicit && m_dt_implicit > 0) {{
                // Use adaptive step size runge-kutta-fehlberg integration for timesteps that are different than implicit integration timestep
                states.data = m_solver.step_runge_kutta_fehlberg(*this, states.data, 0.0, dt);
            }} else {{
                // Solve with Tustin for better accuracy
                states.data = m_solver.step_tustin_fast(*this, states.data, 0.0, dt);
            }}
        }}
        m_dt_prev = dt;


        // Update output
        outputs.data = m_ss.C * states.data + m_ss.D * inputs.data;

        // Update states from outputs to have correct values for dependent states
{update_states}
    }}

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
        uint32_t all;
    }};

    struct StateSpaceMatrices {{
        Eigen::Matrix<double, NUM_STATES, NUM_STATES> A;
        Eigen::Matrix<double, NUM_STATES, NUM_INPUTS> B;
        Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES> C;
        Eigen::Matrix<double, NUM_OUTPUTS, NUM_INPUTS> D;
    }};

    Components components;
    Inputs inputs;
    States states;
    Outputs outputs;
    Switches switches = {{.all = 0}};

  private:
    StateSpaceMatrices calculateStateSpace(Components const& components, Switches switches);

    Integrator<Eigen::Vector<double, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components m_components_DO_NOT_TOUCH;
    Switches m_switches_DO_NOT_TOUCH = {{.all = 0}};
    Eigen::Vector<double, NUM_STATES> m_Bu; // Bu term in "dxdt = Ax + Bu"
    double m_dt_prev = 0;
    double m_dt_implicit = 0;
    // The json file with symbolic intermediate matrices
    std::string m_circuit_json;

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
#include "nlohmann/json.hpp"

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

{class_name}::{class_name}(Components const& c)
    : components(c),
      m_components_DO_NOT_TOUCH(c) {{
    m_ss = calculateStateSpace(components, switches);
}}
''')

    cpp.write(f'''
struct {class_name}_Topology {{
    {class_name}::Components components;
    {class_name}::Switches switches;
    std::unique_ptr<{class_name}::StateSpaceMatrices> state_space;
}};

{class_name}::StateSpaceMatrices {class_name}::calculateStateSpace({class_name}::Components const& components, {class_name}::Switches switches) {{
    static std::vector<{class_name}_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&]({class_name}_Topology const& t) {{
            return t.components == components && t.switches.all == switches.all;
        }});
    if (it != state_space_cache.end()) {{
        return *it->state_space;
    }}
''')


    cpp.write(f'''
    if (m_circuit_json.empty()) {{
        m_circuit_json = rlc2ss::loadTextResource({resource_id});
    }}
    assert(!m_circuit_json.empty());

    // Replace symbolic components with their values before parsing the json
    std::string s = m_circuit_json;
''')

    for component in ss.component_names:
        cpp.write(f"\ts = rlc2ss::replace(s, \"{component}\", std::to_string(components.{component}));\n")

    cpp.write(f'''
    // Parse json for the intermediate matrices
    nlohmann::json j = nlohmann::json::parse(s);
    std::string K1_str = j[std::to_string(switches.all)]["K1"];
    std::string K2_str = j[std::to_string(switches.all)]["K2"];
    std::string A1_str = j[std::to_string(switches.all)]["A1"];
    std::string B1_str = j[std::to_string(switches.all)]["B1"];
    std::string C1_str = j[std::to_string(switches.all)]["C1"];
    std::string D1_str = j[std::to_string(switches.all)]["D1"];

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

    return *topology.state_space;
}}
''')

    circuits = {}

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
        json.dump(circuits, outfile)

    with open(f"{model_name}_matrices.rc", "w") as outfile:
        name = os.path.basename(model_name)
        outfile.write(f'#define {name}_matrices_json {resource_id} \n{name}_matrices_json    TEXT    "{name}_matrices.json"')

    cpp.close()
