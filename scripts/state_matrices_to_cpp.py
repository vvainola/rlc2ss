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
import sympy

@dataclass
class StateSpaceMatrices:
    component_names: list[str]
    states: list[sympy.Symbol]
    inputs: list[sympy.Symbol]
    outputs: list[sympy.Symbol]
    K1: sympy.Matrix
    K2: sympy.Matrix
    A1: sympy.Matrix
    B1: sympy.Matrix
    C1: sympy.Matrix
    D1: sympy.Matrix

def matrices_to_cpp(model_name: str, circuit_combinations: dict[int, StateSpaceMatrices], switches: list[str]):
    hpp = open(f'{model_name}_matrices.hpp', 'w')
    cpp = open(f'{model_name}_matrices.cpp', 'w')
    ss = circuit_combinations[0]

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
    static StateSpaceMatrices calculateStateSpace(Components const& components, Switches switches);

    {class_name}() {{}}
    {class_name}(Components const& c);

    static inline constexpr size_t NUM_INPUTS = {num_inputs};
    static inline constexpr size_t NUM_OUTPUTS = {num_outputs};
    static inline constexpr size_t NUM_STATES = {num_states};
    static inline constexpr size_t NUM_SWITCHES = {num_switches};

    Eigen::Vector<double, NUM_STATES> dxdt(Eigen::Vector<double, NUM_STATES> const& state, double /*t*/) const {{
        return m_ss.A * state + m_Bu;
    }}

    // Implicit Tustin integration is used for timesteps larger than this.
    void setImplicitIntegrationLimit(double dt) {{
        m_solver.enableInverseMatrixCaching(true);
        m_dt_implicit_limit = dt;
    }}

    void step(double dt, Inputs const& inputs_) {{
        inputs.data = inputs_.data;
        // Update state-space matrices if needed
        if (components != m_components_DO_NOT_TOUCH || switches.all != m_switches_DO_NOT_TOUCH.all) {{
{verify_components}
            m_components_DO_NOT_TOUCH = components;
            m_switches_DO_NOT_TOUCH.all = switches.all;
            m_ss = calculateStateSpace(components, switches);
            m_solver.updateJacobian(m_ss.A);
            // Solve one step with backward euler to reduce numerical oscillations
            m_Bu = m_ss.B * inputs.data;
            states.data = m_solver.stepBackwardEuler(*this, states.data, 0.0, dt);
        }} else {{
            m_Bu = m_ss.B * inputs.data;

            if (m_dt_implicit_limit > 0) {{
                // Solve with tustin as multiples of implicit limit and the remaining time with runge-kutta so
                // that the matrix inverses required for implicit integration can be cached for common timesteps
                // and weird small remainders are solved with adaptive integration.
                double multiple = dt / m_dt_implicit_limit;
                if (std::abs(std::round(multiple) - multiple) > 1e-6) {{
                    double dt1 = std::floor(multiple) * m_dt_implicit_limit;
                    double dt2 = (multiple - std::floor(multiple)) * m_dt_implicit_limit;
                    states.data = m_solver.stepTustin(*this, states.data, 0.0, dt1);
                    states.data = m_solver.stepRungeKuttaFehlberg(*this, states.data, 0.0, dt2);
                }} else {{
                    states.data = m_solver.stepTustin(*this, states.data, 0.0, std::round(multiple) * m_dt_implicit_limit);
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
    Inputs m_inputs;
    Integrator<Eigen::Vector<double, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components m_components_DO_NOT_TOUCH;
    Switches m_switches_DO_NOT_TOUCH = {{.all = 0}};
    Eigen::Vector<double, NUM_STATES> m_Bu; // Bu term in "dxdt = Ax + Bu"
    double m_dt_prev = 0;
    double m_dt_implicit_limit = 0;

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

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<{class_name}::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES> const  &K1,
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES> const  &A1,
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_INPUTS> const  &B1,
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES> const &K2,
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES> const &C1,
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_INPUTS> const &D1) {{
    auto ss = std::make_unique<{class_name}::StateSpaceMatrices>();
    ss->A   = K1.partialPivLu().solve(A1);
    ss->B   = K1.partialPivLu().solve(B1);
    ss->C   = (C1 + K2 * ss->A);
    ss->D   = (D1 + K2 * ss->B);
    return ss;
}}

{class_name}::{class_name}(Components const& c)
    : components(c),
      m_components_DO_NOT_TOUCH(c) {{
    m_ss = calculateStateSpace(components, switches);
    m_solver.updateJacobian(m_ss.A);
}}

''')

    for i in sorted(circuit_combinations):
        combination = circuit_combinations[i]
        cpp.write(f'std::unique_ptr<{class_name}::StateSpaceMatrices> calculateStateSpace_{i}({class_name}::Components const& c);\n')

    cpp.write(f'''
struct {class_name}_Topology {{
    {class_name}::Components components;
    {class_name}::Switches switches;
    std::unique_ptr<{class_name}::StateSpaceMatrices> state_space;
}};

{class_name}::StateSpaceMatrices {class_name}::calculateStateSpace({class_name}::Components const& components, {class_name}::Switches switches)
{{
    static std::vector<{class_name}_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&]({class_name}_Topology const& t) {{
        return t.components == components && t.switches.all == switches.all;
    }});
    if (it != state_space_cache.end()) {{
        return *it->state_space;
    }}
    auto state_space = std::make_unique<{class_name}::StateSpaceMatrices>();

    switch (switches.all) {{''')

    for i in sorted(circuit_combinations):
        cpp.write(f'\n\t\tcase {i}: state_space = calculateStateSpace_{i}(components); break;')

    cpp.write(f'''
    default:
        assert(("Invalid switch combination", 0));
    }}
    {class_name}_Topology& topology = state_space_cache.emplace_back({class_name}_Topology{{
        .components = components,
        .switches = switches,
        .state_space = std::move(state_space)}});

    return *topology.state_space;
}}
''')

    write_components = ''
    for component in ss.component_names:
        write_components += f'\tdouble {component} = c.{component};\n'
    for i in sorted(circuit_combinations):
        ss = circuit_combinations[i]
        switch_combination = ''
        for j in range(len(switches)):
            if i & pow(2, j):
                switch_combination += f" {switches[j]}"
        cpp.write(f'''
std::unique_ptr<{class_name}::StateSpaceMatrices> calculateStateSpace_{i}({class_name}::Components const& c) // {switch_combination}
{{
{write_components}
''')

        K1 = str(ss.K1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ' },\n\t\t{') #",".join([str(coeff) for coeff in K1])
        K2 = str(ss.K2).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ' },\n\t\t{') #",".join([str(coeff) for coeff in K2])
        A1 = str(ss.A1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ' },\n\t\t{') #",".join([str(coeff) for coeff in A1])
        B1 = str(ss.B1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ' },\n\t\t{') #",".join([str(coeff) for coeff in B1])
        C1 = str(ss.C1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ' },\n\t\t{') #",".join([str(coeff) for coeff in C1])
        D1 = str(ss.D1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ' },\n\t\t{') #",".join([str(coeff) for coeff in D1])
        cpp.write(f'''
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES> K1 {{\n\t\t{{ {K1} }} }};\n
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES> K2 {{\n\t\t{{ {K2}}} }};\n
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES> A1 {{\n\t\t{{ {A1} }} }};\n
    Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_INPUTS> B1 {{\n\t\t{{ {B1} }} }};\n
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES> C1 {{\n\t\t{{ {C1} }} }};\n
    Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_INPUTS> D1 {{\n\t\t{{ {D1} }} }};

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}}

''')


    cpp.close()
