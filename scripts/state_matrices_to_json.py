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


def render_switch_mask_expr(switches: list[str], prefix: str = "", suffix: str = "") -> str:
    if len(switches) > 0:
        return "0 |" + " |".join(f"\n{TAB*2}(uint64_t{{{prefix}{switch}{suffix}}} << {i})" for i, switch in enumerate(switches))
    return "0"


def render_diode_continuity_methods(
    class_name: str,
    ss: StateSpaceMatrices,
    switches: list[str],
    diodes: list[Diode],
) -> tuple[str, str]:
    external_closed_switches_to_int = render_switch_mask_expr(switches, prefix="switches.", suffix=".output()")
    diode_count = len(diodes)
    instantaneous_outputs_parameter_indent = " " * len(
        f"{class_name}::Outputs calcInstantaneousOutputs("
    )
    release_diodes_parameter_indent = " " * len(
        f"{class_name}::Switches releaseReverseCurrentDiodes("
    )
    resolve_diodes_parameter_indent = " " * len(
        f"{class_name}::Switches resolveDiodeContinuity("
    )

    if diode_count == 0:
        free_functions = render_cxx_block(f"""
            {class_name}::Outputs calcInstantaneousOutputs({class_name}::Components const& components,
            {instantaneous_outputs_parameter_indent}{class_name}::States const& states,
            {instantaneous_outputs_parameter_indent}{class_name}::Inputs const& inputs,
            {instantaneous_outputs_parameter_indent}uint64_t switch_combination) {{
                {class_name}::Outputs instantaneous_outputs;
                auto const& ss = calcStateSpaceMatrices(components, switch_combination);
                instantaneous_outputs.data = ss.C * states.data + ss.D * inputs.data;
                return instantaneous_outputs;
            }}

            uint64_t externalClosedSwitchMask({class_name}::Switches const& switches) {{
                return [[external_closed_switches_to_int]];
            }}

            {class_name}::Switches releaseReverseCurrentDiodes({class_name}::Components const&,
            {release_diodes_parameter_indent}{class_name}::States const&,
            {release_diodes_parameter_indent}{class_name}::Inputs const&,
            {release_diodes_parameter_indent}{class_name}::Switches const& switches) {{
                return switches;
            }}

            {class_name}::Switches resolveDiodeContinuity({class_name}::Components const&,
            {resolve_diodes_parameter_indent}{class_name}::States const&,
            {resolve_diodes_parameter_indent}{class_name}::Inputs const&,
            {resolve_diodes_parameter_indent}{class_name}::Switches const& switches,
            {resolve_diodes_parameter_indent}uint64_t) {{
                return switches;
            }}
        """, indent="", external_closed_switches_to_int=external_closed_switches_to_int)
        methods = ""
        return free_functions, methods

    diode_closed_cases = ""
    diode_controlled_closed_cases = ""
    diode_current_cases = ""
    diode_forward_overdrive_cases = ""
    diode_mask_body = ""
    diode_force_body = ""
    for diode_idx, diode in enumerate(diodes):
        switch_idx = switches.index(diode.switch)
        diode_closed_cases += render_cxx_snippet(f"""
            case {diode_idx}: return switches.{diode.switch}.forcedOutput().value_or(false);
        """, indent=TAB*2)
        diode_controlled_closed_cases += render_cxx_snippet(f"""
            case {diode_idx}: return (external_closed_switch_mask & (uint64_t{{1}} << {switch_idx})) != 0;
        """, indent=TAB*2)
        diode_current_cases += render_cxx_snippet(f"""
            case {diode_idx}: return outputs.{diode.current};
        """, indent=TAB*2)
        pos_node = f"outputs.{diode.pos_node}" if diode.pos_node != "0" else "0.0"
        neg_node = f"outputs.{diode.neg_node}" if diode.neg_node != "0" else "0.0"
        diode_forward_overdrive_cases += render_cxx_snippet(f"""
            case {diode_idx}: return {pos_node} - {neg_node} - inputs.{diode.forward_voltage};
        """, indent=TAB*2)
        diode_mask_body += render_cxx_snippet(f"""
            switch_mask |= ((solver_closed_diode_mask >> {diode_idx}) & uint64_t{{1}}) << {switch_idx};
        """, indent=TAB)
        diode_force_body += render_cxx_snippet(f"""
            switches.{diode.switch}.forceOutput((solver_closed_diode_mask & (uint64_t{{1}} << {diode_idx})) != 0 ? std::optional<bool>{{true}} : std::nullopt);
        """, indent=TAB)

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
                discontinuity = std::max(discontinuity, std::abs(outputs.data[{output_idx}] - states.data[{i}]));
            """, indent=TAB)
            sign_bit_idx += 2

    rendered = render_cxx_block(f"""
        {class_name}::Outputs calcInstantaneousOutputs({class_name}::Components const& components,
        {instantaneous_outputs_parameter_indent}{class_name}::States const& states,
        {instantaneous_outputs_parameter_indent}{class_name}::Inputs const& inputs,
        {instantaneous_outputs_parameter_indent}uint64_t switch_combination) {{
            {class_name}::Outputs instantaneous_outputs;
            // Evaluate the algebraic outputs for an explicit switch mask at t+0.
            // The state vector is not advanced, so any mismatch between inductor
            // output currents and stored inductor states is a real switching
            // discontinuity.
            auto const& ss = calcStateSpaceMatrices(components, switch_combination);
            instantaneous_outputs.data = ss.C * states.data + ss.D * inputs.data;
            return instantaneous_outputs;
        }}

        uint64_t externalClosedSwitchMask({class_name}::Switches const& switches) {{
            // Track each switch's delayed control output, not its possibly
            // solver-closed actual value. This keeps diode zero-crossing
            // force/release events out of external-closure detection while still
            // detecting external closure of a diode switch.
            return [[external_closed_switches_to_int]];
        }}

        bool diodeSolverClosed({class_name}::Switches const& switches, size_t diode_idx) {{
            // True when diode continuity or zero-crossing logic has closed the
            // diode. This excludes closure caused by the external switch control,
            // which is tracked separately by externalClosedSwitchMask().
            switch (diode_idx) {{
        [[diode_closed_cases]]
            default:
                return false;
            }}
        }}

        uint64_t solverClosedDiodeMask({class_name}::Switches const& switches) {{
            uint64_t mask = 0;
            for (size_t diode_idx = 0; diode_idx < {diode_count}; ++diode_idx) {{
                if (diodeSolverClosed(switches, diode_idx)) {{
                    mask |= uint64_t{{1}} << diode_idx;
                }}
            }}
            return mask;
        }}

        uint64_t inductorCurrentSignMask({class_name}::States const& states) {{
            uint64_t mask = 0;
        [[inductor_sign_mask_body]]
            return mask;
        }}

        uint64_t completeTopologyMask(uint64_t external_closed_switch_mask, uint64_t solver_closed_diode_mask) {{
            uint64_t switch_mask = external_closed_switch_mask;
            // The base mask is the external-closed topology. Solver closure
            // can close additional diode switches, but it must not open a diode
            // switch that is already external-closed.
        [[diode_mask_body]]
            return switch_mask;
        }}

        bool diodeExternalClosed(size_t diode_idx, uint64_t external_closed_switch_mask) {{
            switch (diode_idx) {{
        [[diode_controlled_closed_cases]]
            default:
                return false;
            }}
        }}

        double diodeCurrent(size_t diode_idx, {class_name}::Outputs const& outputs) {{
            switch (diode_idx) {{
        [[diode_current_cases]]
            default:
                return 0.0;
            }}
        }}

        double diodeForwardOverdrive(size_t diode_idx,
                                     {class_name}::Outputs const& outputs,
                                     {class_name}::Inputs const& inputs) {{
            // Positive overdrive means an open diode would be forward-biased
            // for this instantaneous solution.
            switch (diode_idx) {{
        [[diode_forward_overdrive_cases]]
            default:
                return 0.0;
            }}
        }}

        double inductorCurrentDiscontinuity({class_name}::Outputs const& outputs,
                                            {class_name}::States const& states) {{
            double discontinuity = 0.0;
            // Inductor current is continuous. The generated state vector stores
            // every inductor current, including dependent inductors, so continuity
            // can be checked without topology-specific knowledge.
        [[inductor_discontinuity_body]]
            return discontinuity;
        }}

        {class_name}::Switches applySolverDiodeMask({class_name}::Switches switches, uint64_t solver_closed_diode_mask) {{
            // Generated diodes are represented as switches in the state-space
            // matrices. Set bits solver-close diodes and clear bits release
            // solver closure. External switch closure remains unchanged.
        [[diode_force_body]]
            return switches;
        }}

        {class_name}::Switches releaseReverseCurrentDiodes({class_name}::Components const& components,
        {release_diodes_parameter_indent}{class_name}::States const& states,
        {release_diodes_parameter_indent}{class_name}::Inputs const& inputs,
        {release_diodes_parameter_indent}{class_name}::Switches const& switches) {{
            uint64_t external_closed_switch_mask = externalClosedSwitchMask(switches);
            uint64_t solver_closed_diode_mask = solverClosedDiodeMask(switches);
            if (solver_closed_diode_mask == 0) {{
                return switches;
            }}

            // A controlled switch closing cannot fix an inductor-current
            // discontinuity by opening diodes; it only gives existing current
            // another path. The full continuity resolver is therefore reserved
            // for switch openings. On a closing-only transition, only release
            // solver-closed diodes carrying reverse current at t+0. Re-evaluate
            // after each release because it can make another diode reverse-biased.
            uint64_t updated_solver_closed_diode_mask = rlc2ss::releaseReverseCurrentDiodeMask(
                solver_closed_diode_mask,
                [&](uint64_t closed_diode_mask) {{
                    uint64_t switch_mask = completeTopologyMask(external_closed_switch_mask, closed_diode_mask);
                    {class_name}::Outputs instantaneous_outputs =
                        calcInstantaneousOutputs(components, states, inputs, switch_mask);
                    uint64_t reverse_current_mask = 0;
                    for (size_t diode_idx = 0; diode_idx < {diode_count}; ++diode_idx) {{
                        uint64_t diode_bit = uint64_t{{1}} << diode_idx;
                        if ((closed_diode_mask & diode_bit) != 0 &&
                            diodeCurrent(diode_idx, instantaneous_outputs) < -rlc2ss::DIODE_CONTINUITY_TOLERANCE) {{
                            reverse_current_mask |= diode_bit;
                        }}
                    }}
                    return reverse_current_mask;
                }});

            return updated_solver_closed_diode_mask != solver_closed_diode_mask
                ? applySolverDiodeMask(switches, updated_solver_closed_diode_mask)
                : switches;
        }}

        {class_name}::Switches resolveDiodeContinuity({class_name}::Components const& components,
        {resolve_diodes_parameter_indent}{class_name}::States const& states,
        {resolve_diodes_parameter_indent}{class_name}::Inputs const& inputs,
        {resolve_diodes_parameter_indent}{class_name}::Switches const& switches,
        {resolve_diodes_parameter_indent}uint64_t previous_topology_mask) {{
            static std::mutex cache_mutex;
            static std::unordered_map<uint64_t, uint64_t> diode_continuity_cache;
            uint64_t external_closed_switch_mask = externalClosedSwitchMask(switches);
            uint64_t initial_solver_closed_diode_mask = solverClosedDiodeMask(switches);

            // Check the diode complementarity part of the candidate solution.
            // Solver-closed diodes may conduct zero or positive current. A diode
            // that is neither solver-closed nor external-closed must not be
            // forward-biased.
            auto diode_complementarity_violation = [&inputs, external_closed_switch_mask](
                uint64_t solver_closed_diode_mask,
                {class_name}::Outputs const& outputs) {{
                double violation = 0.0;
                for (size_t diode_idx = 0; diode_idx < {diode_count}; ++diode_idx) {{
                    if ((solver_closed_diode_mask & (uint64_t{{1}} << diode_idx)) != 0) {{
                        violation = std::max(violation, -diodeCurrent(diode_idx, outputs));
                    }} else if (!diodeExternalClosed(diode_idx, external_closed_switch_mask)) {{
                        violation = std::max(violation, diodeForwardOverdrive(diode_idx, outputs, inputs));
                    }}
                }}
                return violation;
            }};

            double best_attempt_discontinuity = std::numeric_limits<double>::infinity();
            double best_attempt_complementarity_violation = std::numeric_limits<double>::infinity();
            // Evaluate one diode mask at t+0. The state vector is not advanced,
            // so any mismatch between inductor outputs and stored states is the
            // switching discontinuity caused by this topology.
            auto evaluate_mask = [&](uint64_t solver_closed_diode_mask) {{
                uint64_t switch_mask = completeTopologyMask(external_closed_switch_mask, solver_closed_diode_mask);
                {class_name}::Outputs instantaneous_outputs =
                    calcInstantaneousOutputs(components, states, inputs, switch_mask);
                double discontinuity = inductorCurrentDiscontinuity(instantaneous_outputs, states);
                double complementarity_violation = diode_complementarity_violation(solver_closed_diode_mask, instantaneous_outputs);
                best_attempt_discontinuity = std::min(best_attempt_discontinuity, discontinuity);
                best_attempt_complementarity_violation = std::min(best_attempt_complementarity_violation, complementarity_violation);
                return rlc2ss::DiodeContinuityMetrics{{
                    .discontinuity = discontinuity,
                    .complementarity_violation = complementarity_violation,
                }};
            }};

            // Cache only a warm-start set of solver-closed diodes. The same switch
            // transition can need different diodes depending on current direction,
            // so the key includes the inductor-current sign pattern. A cached mask
            // is still fully revalidated before use.
            uint64_t cache_key = 0;
            rlc2ss::hash_combine(cache_key, previous_topology_mask);
            rlc2ss::hash_combine(cache_key, external_closed_switch_mask);
            rlc2ss::hash_combine(cache_key, initial_solver_closed_diode_mask);
            rlc2ss::hash_combine(cache_key, inductorCurrentSignMask(states));

            std::optional<uint64_t> cached_mask;
            {{
                std::scoped_lock<std::mutex> lock(cache_mutex);
                if (auto cached = diode_continuity_cache.find(cache_key); cached != diode_continuity_cache.end()) {{
                    cached_mask = cached->second;
                }}
            }}
            if (cached_mask) {{
                rlc2ss::DiodeContinuityMetrics cached_metrics = evaluate_mask(*cached_mask);
                if (rlc2ss::diodeContinuityValid(cached_metrics, rlc2ss::DIODE_CONTINUITY_TOLERANCE)) {{
                    return applySolverDiodeMask(switches, *cached_mask);
                }}
            }}

            // Fall back to a complete mask search. The helper searches by
            // increasing diode-state changes from the current diode mask, so the
            // common one-diode freewheel case is found before wider combinations
            // are evaluated.
            rlc2ss::DiodeContinuitySelection selection = rlc2ss::selectDiodeContinuityMask(
                {diode_count},
                initial_solver_closed_diode_mask,
                rlc2ss::DIODE_CONTINUITY_TOLERANCE,
                evaluate_mask);

            if (selection.found) {{
                {{
                    std::scoped_lock<std::mutex> lock(cache_mutex);
                    diode_continuity_cache[cache_key] = selection.mask;
                }}
                return applySolverDiodeMask(switches, selection.mask);
            }}

            throw std::runtime_error(std::format(
                "Diode continuity resolver could not find a diode mask satisfying continuity and complementarity; "
                "switch combination {{}} initial diode mask {{}} best residual {{}} best complementarity violation {{}}",
                external_closed_switch_mask,
                initial_solver_closed_diode_mask,
                best_attempt_discontinuity,
                best_attempt_complementarity_violation));
        }}

    """,
        indent="",
        external_closed_switches_to_int=external_closed_switches_to_int,
        inductor_sign_mask_body=inductor_sign_mask_body.rstrip(),
        inductor_discontinuity_body=inductor_discontinuity_body.rstrip(),
        diode_mask_body=diode_mask_body.rstrip(),
        diode_closed_cases=diode_closed_cases.rstrip(),
        diode_controlled_closed_cases=diode_controlled_closed_cases.rstrip(),
        diode_current_cases=diode_current_cases.rstrip(),
        diode_forward_overdrive_cases=diode_forward_overdrive_cases.rstrip(),
        diode_force_body=diode_force_body.rstrip())
    return rendered, ""


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
            if (outputs.{diode.current} < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.{diode.switch}.outputForced()) {{
                events.push(rlc2ss::ZeroCrossingEvent{{
                    .time = rlc2ss::calcZeroCrossingTime(prev_outputs.{diode.current}, outputs.{diode.current}),
                    .event_callback = [this]() {{
                        switches.{diode.switch}.forceOutput(std::nullopt);
                    }}
                }});
            }}
        """)

    diode_continuity_free_functions, diode_continuity_methods = (
        render_diode_continuity_methods(class_name, ss, switches, diodes)
    )

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
            static nlohmann::json const circuit_json = nlohmann::json::parse(
                std::string({model_basename}_matrices_json_hexdump,
                            {model_basename}_matrices_json_hexdump + {model_basename}_matrices_json_hexdump_len));
            assert(circuit_json.contains(std::to_string(switch_combination)));

            // Get the intermediate matrices as string for replacing symbolic components with their values
            std::string s = circuit_json[std::to_string(switch_combination)].dump();
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
            Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES{states_row_major}> K1(rlc2ss::getCommaDelimitedValues(ss.K1).data());
            Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES{states_row_major}> K2(rlc2ss::getCommaDelimitedValues(ss.K2).data());
            Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_STATES{states_row_major}> A1(rlc2ss::getCommaDelimitedValues(ss.A1).data());
            Eigen::Matrix<double, {class_name}::NUM_STATES, {class_name}::NUM_INPUTS{inputs_row_major}> B1(rlc2ss::getCommaDelimitedValues(ss.B1).data());
            Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_STATES{states_row_major}> C1(rlc2ss::getCommaDelimitedValues(ss.C1).data());
            Eigen::Matrix<double, {class_name}::NUM_OUTPUTS, {class_name}::NUM_INPUTS{inputs_row_major}> D1(rlc2ss::getCommaDelimitedValues(ss.D1).data());

            state_space_cache[switch_combination][component_hash] = calcStateSpace(K1, A1, B1, K2, C1, D1);
            return *state_space_cache[switch_combination][component_hash];""", replace_components=replace_components)

    template_context = dict(
        class_name=class_name,
        calc_state_space_parameter_indent=" " * len(
            f"std::unique_ptr<{class_name}::StateSpaceMatrices> calcStateSpace("
        ),
        calc_state_space_matrices_parameter_indent=" " * len(
            f"{class_name}::StateSpaceMatrices const& calcStateSpaceMatrices("
        ),
        model_basename=model_basename,
        num_inputs=len(ss.inputs),
        num_outputs=len(ss.outputs),
        num_states=len(ss.states),
        num_switches=len(switches),
        num_diodes=len(diodes),
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
        diode_continuity_free_functions=diode_continuity_free_functions,
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
