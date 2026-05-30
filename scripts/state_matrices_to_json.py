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


def render_cxx_snippet(source: str, indent: str = TAB, **replacements: str) -> str:
    return render_cxx_block(source, indent, **replacements).strip("\n") + "\n"


def check_for_invalid_names(component_names: list[str]):
    for name in component_names:
        for name2 in component_names:
            if name in name2 and name != name2:
                sys.exit(f"[ERROR]: Component name \"{name}\" cannot be a substring of \"{name2}\".")


def render_switch_mask_expr(switches: list[str], prefix: str = "") -> str:
    if len(switches) > 0:
        return "0 |" + " |".join(f"\n{TAB*2}({prefix}{switch} << {i})" for i, switch in enumerate(switches))
    return "0"


def render_diode_continuity_methods(
    class_name: str,
    ss: StateSpaceMatrices,
    switches: list[str],
    diodes: list[Diode],
) -> str:
    diode_output_names = {diode.switch for diode in diodes}
    controlled_switches = [switch for switch in switches if switch not in diode_output_names]
    controlled_switches_to_int = render_switch_mask_expr(controlled_switches, prefix="switches.")
    diode_count = len(diodes)
    if diode_count == 0:
        return render_cxx_block(f"""
            void {class_name}::resolveDiodeContinuity() {{
                m_last_switch_mask = switches.all();
            }}

            {class_name}::Outputs {class_name}::calcInstantaneousOutputs(uint64_t switch_combination) {{
                Outputs instantaneous_outputs;
                StateSpaceMatrices ss = calcStateSpaceMatrices(switch_combination);
                instantaneous_outputs.data = ss.C * states.data + ss.D * inputs.data;
                return instantaneous_outputs;
            }}

            uint64_t {class_name}::controlledSwitchMask() const {{
                return [[controlled_switches_to_int]];
            }}

            uint64_t {class_name}::closedDiodeMask() const {{
                return 0;
            }}

            uint64_t {class_name}::inductorCurrentSignMask() const {{
                return 0;
            }}

            uint64_t {class_name}::switchMaskWithClosedDiodes(uint64_t base_switch_mask, uint64_t) const {{
                return base_switch_mask;
            }}

            bool {class_name}::diodeClosed(size_t) const {{
                return false;
            }}

            double {class_name}::diodeCurrent(size_t, Outputs const&) const {{
                return 0.0;
            }}

            double {class_name}::diodeForwardOverdrive(size_t, Outputs const&) const {{
                return 0.0;
            }}

            double {class_name}::inductorCurrentDiscontinuity(Outputs const&) const {{
                return 0.0;
            }}

            void {class_name}::forceClosedDiodeMask(uint64_t) {{
            }}

            void {class_name}::releaseReverseCurrentDiodes() {{
                m_last_switch_mask = switches.all();
            }}
        """, indent="", controlled_switches_to_int=controlled_switches_to_int)

    diode_closed_cases = ""
    diode_current_cases = ""
    diode_forward_overdrive_cases = ""
    diode_mask_body = ""
    diode_force_cases = ""
    for i, diode in enumerate(diodes):
        diode_closed_cases += render_cxx_snippet(f"""
            case {i}: return switches.{diode.switch};
        """, indent=TAB*2)
        diode_current_cases += render_cxx_snippet(f"""
            case {i}: return outputs_.{diode.current};
        """, indent=TAB*2)
        pos_node = f"outputs_.{diode.pos_node}" if diode.pos_node != "0" else "0.0"
        neg_node = f"outputs_.{diode.neg_node}" if diode.neg_node != "0" else "0.0"
        diode_forward_overdrive_cases += render_cxx_snippet(f"""
            case {i}: return {pos_node} - {neg_node} - inputs.{diode.forward_voltage};
        """, indent=TAB*2)
        switch_idx = switches.index(diode.switch)
        diode_mask_body += render_cxx_snippet(f"""
            if ((closed_diode_mask & (uint64_t{{1}} << {i})) != 0) {{
                switch_mask |= uint64_t{{1}} << {switch_idx};
            }} else {{
                switch_mask &= ~(uint64_t{{1}} << {switch_idx});
            }}
        """, indent=TAB)
        diode_force_cases += render_cxx_snippet(f"""
            case {i}:
                switches.{diode.switch}.forceOutput((closed_diode_mask & (uint64_t{{1}} << {i})) != 0 ? std::optional<bool>{{true}} : std::nullopt);
                break;
        """, indent=TAB*2)

    inductor_sign_mask_body = ""
    inductor_discontinuity_body = ""
    sign_bit_idx = 0
    for i, state in enumerate(ss.states):
        state_name = str(state)
        if state_name.startswith("I_L"):
            inductor_sign_mask_body += render_cxx_snippet(f"""
                if (states.{state_name} > 0.0) {{
                    mask |= uint64_t{{1}} << {sign_bit_idx};
                }} else if (states.{state_name} < 0.0) {{
                    mask |= uint64_t{{1}} << {sign_bit_idx + 1};
                }}
            """, indent=TAB)
            output_idx = ss.outputs.index(state)
            inductor_discontinuity_body += render_cxx_snippet(f"""
                discontinuity = std::max(discontinuity, std::abs(outputs_.data[{output_idx}] - states.data[{i}]));
            """, indent=TAB)
            sign_bit_idx += 2

    return render_cxx_block(f"""
        namespace {{
        uint64_t mixContinuityCacheKey(uint64_t seed, uint64_t value) {{
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            return seed;
        }}
        }} // namespace

        {class_name}::Outputs {class_name}::calcInstantaneousOutputs(uint64_t switch_combination) {{
            Outputs instantaneous_outputs;
            // Evaluate the algebraic outputs for an explicit switch mask at t+0.
            // The state vector is not advanced, so any mismatch between inductor
            // output currents and stored inductor states is a real switching
            // discontinuity.
            StateSpaceMatrices ss = calcStateSpaceMatrices(switch_combination);
            instantaneous_outputs.data = ss.C * states.data + ss.D * inputs.data;
            return instantaneous_outputs;
        }}

        uint64_t {class_name}::controlledSwitchMask() const {{
            return [[controlled_switches_to_int]];
        }}

        uint64_t {class_name}::closedDiodeMask() const {{
            uint64_t mask = 0;
            for (size_t diode_idx = 0; diode_idx < {diode_count}; ++diode_idx) {{
                if (diodeClosed(diode_idx)) {{
                    mask |= uint64_t{{1}} << diode_idx;
                }}
            }}
            return mask;
        }}

        uint64_t {class_name}::inductorCurrentSignMask() const {{
            uint64_t mask = 0;
        [[inductor_sign_mask_body]]
            return mask;
        }}

        uint64_t {class_name}::switchMaskWithClosedDiodes(uint64_t base_switch_mask, uint64_t closed_diode_mask) const {{
            uint64_t switch_mask = base_switch_mask;
        [[diode_mask_body]]
            return switch_mask;
        }}

        bool {class_name}::diodeClosed(size_t diode_idx) const {{
            switch (diode_idx) {{
        [[diode_closed_cases]]
            default:
                return false;
            }}
        }}

        double {class_name}::diodeCurrent(size_t diode_idx, Outputs const& outputs_) const {{
            switch (diode_idx) {{
        [[diode_current_cases]]
            default:
                return 0.0;
            }}
        }}

        double {class_name}::diodeForwardOverdrive(size_t diode_idx, Outputs const& outputs_) const {{
            // Positive overdrive means an open diode would be forward-biased
            // for this instantaneous solution.
            switch (diode_idx) {{
        [[diode_forward_overdrive_cases]]
            default:
                return 0.0;
            }}
        }}

        double {class_name}::inductorCurrentDiscontinuity(Outputs const& outputs_) const {{
            double discontinuity = 0.0;
            // Inductor current is continuous. The generated state vector stores
            // every inductor current, including dependent inductors, so continuity
            // can be checked without topology-specific knowledge.
        [[inductor_discontinuity_body]]
            return discontinuity;
        }}

        void {class_name}::forceClosedDiodeMask(uint64_t closed_diode_mask) {{
            // Generated diodes are represented as switches in the state-space
            // matrices. Setting a bit here forces that diode closed;
            // clearing a bit releases the force, which leaves generated diode
            // outputs open until diode zero-crossing logic forces them closed.
            for (size_t diode_idx = 0; diode_idx < {diode_count}; ++diode_idx) {{
                switch (diode_idx) {{
        [[diode_force_cases]]
                default:
                    break;
                }}
            }}
        }}

        void {class_name}::releaseReverseCurrentDiodes() {{
            constexpr double DIODE_CONTINUITY_TOLERANCE = 1e-9;
            uint64_t current_switch_mask = switches.all();
            uint64_t closed_diode_mask = closedDiodeMask();
            if (closed_diode_mask == 0) {{
                m_last_switch_mask = current_switch_mask;
                return;
            }}

            // A controlled switch closing cannot fix an inductor-current
            // discontinuity by opening diodes; it only gives existing current
            // another path. The full continuity resolver is therefore reserved
            // for switch openings. On a closing-only transition, only release
            // forced diodes that are already carrying reverse current at t+0.
            uint64_t switch_mask = switchMaskWithClosedDiodes(current_switch_mask, closed_diode_mask);
            Outputs instantaneous_outputs = calcInstantaneousOutputs(switch_mask);
            uint64_t updated_closed_diode_mask = closed_diode_mask;
            for (size_t diode_idx = 0; diode_idx < {diode_count}; ++diode_idx) {{
                uint64_t diode_bit = uint64_t{{1}} << diode_idx;
                if ((closed_diode_mask & diode_bit) != 0 &&
                    diodeCurrent(diode_idx, instantaneous_outputs) < -DIODE_CONTINUITY_TOLERANCE) {{
                    updated_closed_diode_mask &= ~diode_bit;
                }}
            }}

            if (updated_closed_diode_mask != closed_diode_mask) {{
                forceClosedDiodeMask(updated_closed_diode_mask);
            }}
            m_last_switch_mask = switchMaskWithClosedDiodes(current_switch_mask, updated_closed_diode_mask);
        }}

        void {class_name}::resolveDiodeContinuity() {{
            constexpr double DIODE_CONTINUITY_TOLERANCE = 1e-9;
            uint64_t current_switch_mask = switches.all();
            uint64_t initial_closed_diode_mask = closedDiodeMask();

            // Check the diode complementarity part of the candidate solution.
            // Closed diodes may conduct zero or positive current. Open diodes
            // must not be forward-biased.
            auto diode_complementarity_violation = [this](uint64_t closed_diode_mask, Outputs const& outputs_) {{
                double violation = 0.0;
                for (size_t diode_idx = 0; diode_idx < {diode_count}; ++diode_idx) {{
                    if ((closed_diode_mask & (uint64_t{{1}} << diode_idx)) != 0) {{
                        violation = std::max(violation, -diodeCurrent(diode_idx, outputs_));
                    }} else {{
                        violation = std::max(violation, diodeForwardOverdrive(diode_idx, outputs_));
                    }}
                }}
                return violation;
            }};

            double best_attempt_discontinuity = std::numeric_limits<double>::infinity();
            double best_attempt_complementarity_violation = std::numeric_limits<double>::infinity();
            // Evaluate one diode mask at t+0. The state vector is not advanced,
            // so any mismatch between inductor outputs and stored states is the
            // switching discontinuity caused by this topology.
            auto evaluate_mask = [&](uint64_t closed_diode_mask) {{
                uint64_t switch_mask = switchMaskWithClosedDiodes(current_switch_mask, closed_diode_mask);
                Outputs instantaneous_outputs = calcInstantaneousOutputs(switch_mask);
                double discontinuity = inductorCurrentDiscontinuity(instantaneous_outputs);
                double complementarity_violation = diode_complementarity_violation(closed_diode_mask, instantaneous_outputs);
                best_attempt_discontinuity = std::min(best_attempt_discontinuity, discontinuity);
                best_attempt_complementarity_violation = std::min(best_attempt_complementarity_violation, complementarity_violation);
                return rlc2ss::DiodeContinuityMetrics{{
                    .discontinuity = discontinuity,
                    .complementarity_violation = complementarity_violation,
                }};
            }};

            // Cache only a warm-start set of closed diodes. The same switch
            // transition can need different diodes depending on current direction,
            // so the key includes the inductor-current sign pattern. A cached mask
            // is still fully revalidated before use.
            uint64_t cache_key = 0;
            cache_key = mixContinuityCacheKey(cache_key, m_last_switch_mask);
            cache_key = mixContinuityCacheKey(cache_key, current_switch_mask);
            cache_key = mixContinuityCacheKey(cache_key, initial_closed_diode_mask);
            cache_key = mixContinuityCacheKey(cache_key, inductorCurrentSignMask());

            if (auto cached = m_diode_continuity_cache.find(cache_key); cached != m_diode_continuity_cache.end()) {{
                rlc2ss::DiodeContinuityMetrics cached_metrics = evaluate_mask(cached->second);
                if (rlc2ss::diodeContinuityValid(cached_metrics, DIODE_CONTINUITY_TOLERANCE)) {{
                    uint64_t cached_switch_mask = switchMaskWithClosedDiodes(current_switch_mask, cached->second);
                    forceClosedDiodeMask(cached->second);
                    m_last_switch_mask = cached_switch_mask;
                    return;
                }}
            }}

            // Fall back to a complete mask search. The helper searches by
            // increasing diode-state changes from the current diode mask, so the
            // common one-diode freewheel case is found before wider combinations
            // are evaluated.
            rlc2ss::DiodeContinuitySelection selection = rlc2ss::selectDiodeContinuityMask(
                {diode_count},
                initial_closed_diode_mask,
                DIODE_CONTINUITY_TOLERANCE,
                evaluate_mask);

            if (selection.found) {{
                uint64_t final_switch_mask = switchMaskWithClosedDiodes(current_switch_mask, selection.mask);
                forceClosedDiodeMask(selection.mask);
                m_diode_continuity_cache[cache_key] = selection.mask;
                m_last_switch_mask = final_switch_mask;
                return;
            }}

            throw std::runtime_error(std::format(
                "Diode continuity resolver could not find a diode mask satisfying continuity and complementarity; "
                "switch combination {{}} initial diode mask {{}} best residual {{}} best complementarity violation {{}}",
                current_switch_mask,
                initial_closed_diode_mask,
                best_attempt_discontinuity,
                best_attempt_complementarity_violation));
        }}
    """,
        indent="",
        controlled_switches_to_int=controlled_switches_to_int,
        inductor_sign_mask_body=inductor_sign_mask_body.rstrip(),
        inductor_discontinuity_body=inductor_discontinuity_body.rstrip(),
        diode_mask_body=diode_mask_body.rstrip(),
        diode_closed_cases=diode_closed_cases.rstrip(),
        diode_current_cases=diode_current_cases.rstrip(),
        diode_forward_overdrive_cases=diode_forward_overdrive_cases.rstrip(),
        diode_force_cases=diode_force_cases.rstrip())


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
    switches_to_int = render_switch_mask_expr(switches)
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

    diode_continuity_methods = render_diode_continuity_methods(class_name, ss, switches, diodes)

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
            return *state_space_cache[switch_combination][component_hash];""", values_list=values_list)
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
            assert(m_circuit_json.contains(std::to_string(switch_combination)));

            // Get the intermediate matrices as string for replacing symbolic components with their values
            std::string s = m_circuit_json[std::to_string(switch_combination)].dump();
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
            return *state_space_cache[switch_combination][component_hash];""", replace_components=replace_components)

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
        diode_continuity_methods=diode_continuity_methods,
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
