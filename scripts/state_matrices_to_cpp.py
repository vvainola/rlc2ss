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

import math
def matrices_to_cpp(filename, circuit_combinations, switches):

    f = open(filename, 'w')
    (component_names, states, inputs, outputs, K1, K2, A1, B1, C1, D1) = circuit_combinations[0]

    component_fields = [f'\tdouble {c};' for c in component_names]
    component_fields = "\n".join(component_fields)
    f.write(f'''
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <bitset>

struct Components
{{
{component_fields}
}};
''')

    state_fields = ['\t' + str(state) for state in states]
    state_fields = ",\n".join(state_fields)
    f.write(f'''
enum class State
{{
{state_fields},
\tNUM_STATES
}};
''')

    inputs = ['\t' + str(input) for input in inputs]
    inputs = ",\n".join(inputs)
    f.write(f'''
enum class Input
{{
{inputs},
\tNUM_INPUTS
}};
''')

    outputs = ['\t' + str(output) for output in outputs]
    outputs = ",\n".join(outputs)
    f.write(f'''
enum class Output
{{
{outputs},
\tNUM_OUTPUTS
}};

struct StateSpaceMatrices
{{
    Eigen::Matrix<double, int(State::NUM_STATES), int(State::NUM_STATES)> A;
    Eigen::Matrix<double, int(State::NUM_STATES), int(Input::NUM_INPUTS)> B;
    Eigen::Matrix<double, int(Output::NUM_OUTPUTS), int(State::NUM_STATES)> C;
    Eigen::Matrix<double, int(Output::NUM_OUTPUTS), int(Input::NUM_INPUTS)> D;
}};
''')
    if len(switches) > 0:
        switches_str = ['\t' + str(switch) for switch in switches]
        switches_str = ",\n".join(switches_str)
        f.write(f'''
enum Switch
{{
{switches_str},
\tNUM_SWITCHES
}};

''')
    else:
            f.write(f'''
enum Switch
{{
\tNO_SWITCHES,
\tNUM_SWITCHES
}};

''')
    for i, combination in enumerate(circuit_combinations):
        f.write(f'StateSpaceMatrices calculateStateSpace_{i}(Components const& c);\n')

    f.write(f'''
using SwitchPositions = std::bitset<int(Switch::NUM_SWITCHES)>;

StateSpaceMatrices calculateStateSpace(Components const& components, uint64_t switches = 0)
{{
''')
    f.write(f'\n\tswitch (switches) {{')

    for i, combination in enumerate(circuit_combinations):
        f.write(f'\n\t\tcase {i}: return calculateStateSpace_{i}(components);')

    f.write(f'''
    default:
        assert(0);
    }}

    return {{}};
}}
''')
    write_components = ''
    for component in component_names:
        write_components += f'\tdouble {component} = c.{component};\n'
    
    for i, combination in enumerate(circuit_combinations):
        switch_combination = ''
        for j in range(len(switches)):
            if i & pow(2, j):
                switch_combination += f" {switches[j]}"
        f.write(f'''
StateSpaceMatrices calculateStateSpace_{i}(Components const& c) // {switch_combination}
{{
{write_components}

    Eigen::Matrix<double, int(State::NUM_STATES), int(State::NUM_STATES)> K1;
    Eigen::Matrix<double, int(State::NUM_STATES), int(State::NUM_STATES)> A1;
    Eigen::Matrix<double, int(State::NUM_STATES), int(Input::NUM_INPUTS)> B1;
    Eigen::Matrix<double, int(Output::NUM_OUTPUTS), int(State::NUM_STATES)> K2;
    Eigen::Matrix<double, int(Output::NUM_OUTPUTS), int(State::NUM_STATES)> C1;
    Eigen::Matrix<double, int(Output::NUM_OUTPUTS), int(Input::NUM_INPUTS)> D1;
        ''')
        
        (component_names, states, inputs, outputs, K1, K2, A1, B1, C1, D1) = combination
        K1 = str(K1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',\n\t\t\t') #",".join([str(coeff) for coeff in K1])
        K2 = str(K2).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',\n\t\t\t') #",".join([str(coeff) for coeff in K2])
        A1 = str(A1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',\n\t\t\t') #",".join([str(coeff) for coeff in A1])
        B1 = str(B1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',\n\t\t\t') #",".join([str(coeff) for coeff in B1])
        C1 = str(C1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',\n\t\t\t') #",".join([str(coeff) for coeff in C1])
        D1 = str(D1).replace('Matrix([[', '').replace(']])', '').replace('[', '').replace('],', ',\n\t\t\t') #",".join([str(coeff) for coeff in D1])
        f.write(f'''
    K1 <<\n\t\t\t {K1};
    K2 <<\n\t\t\t {K2};
    A1 <<\n\t\t\t {A1};
    B1 <<\n\t\t\t {B1};
    C1 <<\n\t\t\t {C1};
    D1 <<\n\t\t\t {D1};

    StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}}

''')
    
    f.close()
