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
import textwrap
import sympy
from jinja2 import Environment, FileSystemLoader

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
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def render_template(template_name: str, **context) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        keep_trailing_newline=True,
        variable_start_string="[[",
        variable_end_string="]]",
        autoescape=False,
    )
    return env.get_template(template_name).render(**context)


def render_cxx_block(source: str, indent: str = TAB, **replacements: str) -> str:
    block = textwrap.dedent(source)
    for key, value in replacements.items():
        block = block.replace(f"[[{key}]]", value)
    return textwrap.indent(block, indent)


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
    components_compare = " &&\n".join([f'{TAB*2}{str(component)} == other.{str(component)}' for component in ss.component_names])
    components_hash = "\n".join(f'{TAB}rlc2ss::hash_combine(seed, {str(component)});' for component in ss.component_names)
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

    # Sort diodes by their name for deterministic ordering
    diodes.sort(key=lambda d: d.name)
    diode_zero_crossing_events = ""
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

        diode_zero_crossing_events += render_cxx_block(f"""
            // Diode {diode.name}
            double V_{diode.name} = {pos_node} - {neg_node};
            if (V_{diode.name} > inputs.{diode.forward_voltage} && !switches.{diode.switch}) {{
                double V_{diode.name}_prev = {prev_pos_node} - {prev_neg_node};
                events.push(rlc2ss::ZeroCrossingEvent{{
                    .time = rlc2ss::calcZeroCrossingTime(V_{diode.name}_prev, V_{diode.name}),
                    .event_callback = [this]() {{
                        switches.{diode.switch}.forceOutput(true);
                    }}
                }});
            }}
            if (outputs.{diode.current} < 0 && switches.{diode.switch}.outputForced()) {{
                events.push(rlc2ss::ZeroCrossingEvent{{
                    .time = rlc2ss::calcZeroCrossingTime(prev_outputs.{diode.current}, outputs.{diode.current}),
                    .event_callback = [this]() {{
                        switches.{diode.switch}.forceOutput(std::nullopt);
                    }}
                }});
            }}
        """)

    inductor_saturation_indices = ""
    for component in ss.component_names:
         if component.startswith('L'):
            inductor_saturation_indices += render_cxx_block(f"""
                if (inductor == &components.{component}) {{
                    i_L_output_idx = {ss.outputs.index(sympy.Symbol(f'I_{component}'))};
                }}
            """).rstrip("\n")

    include_json_header = f'#include "{model_basename}_matrices_json.h"' if not dynamic else ''
    if dynamic:
        values_list = "".join(f'{TAB}{{"{component}", components.{component}}},\n' for component in ss.component_names)
        update_state_space_matrices_body = render_cxx_block(f"""\
            std::string netlist = "{netlist}";

            // Cache symbolic intermediate matrices per switch combination
            static std::unordered_map<uint64_t, rlc2ss::SymbolicStateSpace> symbolic_cache;
            if (!symbolic_cache.contains(switch_combination)) {{
                symbolic_cache[switch_combination] = rlc2ss::formStateSpaceMatrices(netlist, switch_combination);
            }}
            rlc2ss::SymbolicStateSpace const& symbolic_ss = symbolic_cache[switch_combination];

            // Substitute component values into cached symbolic matrices
            std::unordered_map<std::string, double> values{{
            [[values_list]]}};
            Eigen::MatrixXd K1 = rlc2ss::evaluate(symbolic_ss.K1, values);
            Eigen::MatrixXd K2 = rlc2ss::evaluate(symbolic_ss.K2, values);
            Eigen::MatrixXd A1 = rlc2ss::evaluate(symbolic_ss.A1, values);
            Eigen::MatrixXd B1 = rlc2ss::evaluate(symbolic_ss.B1, values);
            Eigen::MatrixXd C1 = rlc2ss::evaluate(symbolic_ss.C1, values);
            Eigen::MatrixXd D1 = rlc2ss::evaluate(symbolic_ss.D1, values);

            state_space_cache[switch_combination][component_hash] = calcStateSpace(K1, A1, B1, K2, C1, D1);
            m_ss = *state_space_cache[switch_combination][component_hash];""", values_list=values_list)
    else:
        replace_components = "".join(
            f's = rlc2ss::replace(s, "{component}", std::format("({{}})", components.{component}));\n'
            for component in ss.component_names
        )
        # The row major template parameter can only be specified if there is more than 1 column
        states_row_major = f", Eigen::RowMajor" if len(ss.states) > 1 else ""
        inputs_row_major = f", Eigen::RowMajor" if len(ss.inputs) > 1 else ""
        update_state_space_matrices_body = render_cxx_block(f"""
            if (m_circuit_json.empty()) {{
                m_circuit_json = nlohmann::json::parse(std::string({model_basename}_matrices_json_hexdump, {model_basename}_matrices_json_hexdump + {model_basename}_matrices_json_hexdump_len));
            }}
            assert(m_circuit_json.contains(std::to_string(switches.all())));

            // Get the intermediate matrices as string for replacing symbolic components with their values
            std::string s = m_circuit_json[std::to_string(switches.all())].dump();
            [[replace_components]]
            // Parse json for the intermediate matrices
            nlohmann::json j = nlohmann::json::parse(s);
            rlc2ss::StateSpaceMatrices ss = {{
                .K1 = j["K1"],
                .K2 = j["K2"],
                .A1 = j["A1"],
                .B1 = j["B1"],
                .C1 = j["C1"],
                .D1 = j["D1"],
            }};
            // Create eigen matrices
            Eigen::Matrix<double, NUM_STATES, NUM_STATES{states_row_major}> K1(rlc2ss::getCommaDelimitedValues(ss.K1).data());
            Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES{states_row_major}> K2(rlc2ss::getCommaDelimitedValues(ss.K2).data());
            Eigen::Matrix<double, NUM_STATES, NUM_STATES{states_row_major}> A1(rlc2ss::getCommaDelimitedValues(ss.A1).data());
            Eigen::Matrix<double, NUM_STATES, NUM_INPUTS{inputs_row_major}> B1(rlc2ss::getCommaDelimitedValues(ss.B1).data());
            Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES{states_row_major}> C1(rlc2ss::getCommaDelimitedValues(ss.C1).data());
            Eigen::Matrix<double, NUM_OUTPUTS, NUM_INPUTS{inputs_row_major}> D1(rlc2ss::getCommaDelimitedValues(ss.D1).data());

            state_space_cache[switch_combination][component_hash] = calcStateSpace(K1, A1, B1, K2, C1, D1);
            m_ss = *state_space_cache[switch_combination][component_hash];""", replace_components=replace_components)

    template_context = dict(
        class_name=class_name,
        model_basename=model_basename,
        num_inputs=len(ss.inputs),
        num_outputs=len(ss.outputs),
        num_states=len(ss.states),
        num_switches=len(switches),
        components_list=components_list,
        components_compare=components_compare,
        components_hash=components_hash,
        verify_components=verify_components,
        states_list=states_list,
        inputs_list=inputs_list,
        outputs_list=outputs_list,
        switches_list=switches_list,
        update_states=update_states,
        switches_to_int=switches_to_int,
        switches_min_delay=switches_min_delay,
        switches_step=switches_step,
        include_json_header=include_json_header,
        diode_zero_crossing_events=diode_zero_crossing_events,
        inductor_saturation_indices=inductor_saturation_indices,
        update_state_space_matrices_body=update_state_space_matrices_body,
    )
    hpp.write(render_template("model_matrices.hpp.j2", **template_context).replace('\t', TAB))
    hpp.close()
    cpp.write(render_template("model_matrices.cpp.j2", **template_context).replace('\t', TAB))
    cpp.close()
    return

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
