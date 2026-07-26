
#include "diode_bridge_matrices.hpp"
#include "diode_continuity.hpp"
#include "rlc2ss.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <mutex>
#include <format>
#include <memory>
#include <stdexcept>


#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

inline constexpr int MAX_ZERO_CROSS_EVENTS = 100;

namespace {

std::unique_ptr<Model_diode_bridge::StateSpaceMatrices> calcStateSpace(Eigen::MatrixXd const& K1,
                                                                       Eigen::MatrixXd const& A1,
                                                                       Eigen::MatrixXd const& B1,
                                                                       Eigen::MatrixXd const& K2,
                                                                       Eigen::MatrixXd const& C1,
                                                                       Eigen::MatrixXd const& D1) {
    auto ss = std::make_unique<Model_diode_bridge::StateSpaceMatrices>();
    auto lu = K1.partialPivLu();
    Eigen::MatrixXd A = lu.solve(A1);
    Eigen::MatrixXd B = lu.solve(B1);
    ss->A = A;
    ss->B = B;
    ss->C = (C1 + K2 * A);
    ss->D = (D1 + K2 * B);
    return ss;
}

Model_diode_bridge::StateSpaceMatrices const& calcStateSpaceMatrices(Model_diode_bridge::Components const& components,
                                                                     uint64_t switch_combination) {
    static std::mutex            cache_mutex;
    std::scoped_lock<std::mutex> lock(cache_mutex);

    using StateSpaceMap = std::unordered_map<uint64_t, std::unique_ptr<Model_diode_bridge::StateSpaceMatrices>>;
    static std::unordered_map<uint64_t, StateSpaceMap> state_space_cache;
    uint64_t component_hash = components.hash();
    if (state_space_cache.contains(switch_combination)) {
        std::unordered_map<uint64_t, std::unique_ptr<Model_diode_bridge::StateSpaceMatrices>>& cache = state_space_cache.at(switch_combination);
        auto it = cache.find(component_hash);
        if (it != cache.end()) {
            return *it->second;
        }
    }
    std::string netlist = "D_p_a _net0 N_dc_p \nD_p_b _net1 N_dc_p \nD_p_c _net2 N_dc_p \nD_n_a N_dc_n _net0 \nD_n_b N_dc_n _net1 \nD_n_c N_dc_n _net2 \nV_a _net3 0 DC 1 \nV_b _net4 0 DC 1 \nV_c _net5 0 DC 1 \nR_a _net0 _net6 50; \nR_b _net1 _net7 50; \nR_c _net2 _net8 50; \nR_load N_dc_p N_dc_n 50; \nR_dc N_dc_p _net9 50; \nC_dc _net9 N_dc_n 1P \nL_a _net3 _net6 1e-9; \nL_b _net4 _net7 1e-9; \nL_c _net5 _net8 1e-9; ";

    // Cache symbolic intermediate matrices per switch combination
    static std::unordered_map<uint64_t, rlc2ss::SymbolicStateSpace> symbolic_cache;
    if (!symbolic_cache.contains(switch_combination)) {
        symbolic_cache[switch_combination] = rlc2ss::formStateSpaceMatrices(netlist, switch_combination);
    }
    rlc2ss::SymbolicStateSpace const& symbolic_ss = symbolic_cache[switch_combination];

    // Substitute component values into cached symbolic matrices
    std::unordered_map<std::string, double> values{
        {"C_dc", components.C_dc},
        {"L_a", components.L_a},
        {"L_b", components.L_b},
        {"L_c", components.L_c},
        {"R_D_n_a", components.R_D_n_a},
        {"R_D_n_b", components.R_D_n_b},
        {"R_D_n_c", components.R_D_n_c},
        {"R_D_p_a", components.R_D_p_a},
        {"R_D_p_b", components.R_D_p_b},
        {"R_D_p_c", components.R_D_p_c},
        {"R_a", components.R_a},
        {"R_b", components.R_b},
        {"R_c", components.R_c},
        {"R_dc", components.R_dc},
        {"R_load", components.R_load},
    };
    Eigen::MatrixXd K1 = rlc2ss::evaluate(symbolic_ss.K1, values);
    Eigen::MatrixXd K2 = rlc2ss::evaluate(symbolic_ss.K2, values);
    Eigen::MatrixXd A1 = rlc2ss::evaluate(symbolic_ss.A1, values);
    Eigen::MatrixXd B1 = rlc2ss::evaluate(symbolic_ss.B1, values);
    Eigen::MatrixXd C1 = rlc2ss::evaluate(symbolic_ss.C1, values);
    Eigen::MatrixXd D1 = rlc2ss::evaluate(symbolic_ss.D1, values);

    state_space_cache[switch_combination][component_hash] = calcStateSpace(K1, A1, B1, K2, C1, D1);
    return *state_space_cache[switch_combination][component_hash];
}


Model_diode_bridge::Outputs calcInstantaneousOutputs(Model_diode_bridge::Components const& components,
                                                     Model_diode_bridge::States const& states,
                                                     Model_diode_bridge::Inputs const& inputs,
                                                     uint64_t switch_combination) {
    Model_diode_bridge::Outputs instantaneous_outputs;
    // Evaluate the algebraic outputs for an explicit switch mask at t+0.
    // The state vector is not advanced, so any mismatch between inductor
    // output currents and stored inductor states is a real switching
    // discontinuity.
    auto const& ss = calcStateSpaceMatrices(components, switch_combination);
    instantaneous_outputs.data = ss.C * states.data + ss.D * inputs.data;
    return instantaneous_outputs;
}

uint64_t externalClosedSwitchMask(Model_diode_bridge::Switches const& switches) {
    // Track each switch's delayed control output, not its possibly
    // solver-closed actual value. This keeps diode zero-crossing
    // force/release events out of external-closure detection while still
    // detecting external closure of a diode switch.
    return 0 |
        (uint64_t{switches.S_D_n_a.output()} << 0) |
        (uint64_t{switches.S_D_n_b.output()} << 1) |
        (uint64_t{switches.S_D_n_c.output()} << 2) |
        (uint64_t{switches.S_D_p_a.output()} << 3) |
        (uint64_t{switches.S_D_p_b.output()} << 4) |
        (uint64_t{switches.S_D_p_c.output()} << 5);
}

bool diodeSolverClosed(Model_diode_bridge::Switches const& switches, size_t diode_idx) {
    // True when diode continuity or zero-crossing logic has closed the
    // diode. This excludes closure caused by the external switch control,
    // which is tracked separately by externalClosedSwitchMask().
    switch (diode_idx) {
        case 0: return switches.S_D_n_a.forcedOutput().value_or(false);
        case 1: return switches.S_D_n_b.forcedOutput().value_or(false);
        case 2: return switches.S_D_n_c.forcedOutput().value_or(false);
        case 3: return switches.S_D_p_a.forcedOutput().value_or(false);
        case 4: return switches.S_D_p_b.forcedOutput().value_or(false);
        case 5: return switches.S_D_p_c.forcedOutput().value_or(false);
    default:
        return false;
    }
}

uint64_t solverClosedDiodeMask(Model_diode_bridge::Switches const& switches) {
    uint64_t mask = 0;
    for (size_t diode_idx = 0; diode_idx < 6; ++diode_idx) {
        if (diodeSolverClosed(switches, diode_idx)) {
            mask |= uint64_t{1} << diode_idx;
        }
    }
    return mask;
}

uint64_t inductorCurrentSignMask(Model_diode_bridge::States const& states) {
    uint64_t mask = 0;
    if (states.I_L_a > 0.0) {
        mask |= uint64_t{1} << 0;
    } else if (states.I_L_a < 0.0) {
        mask |= uint64_t{1} << 1;
    }
    if (states.I_L_b > 0.0) {
        mask |= uint64_t{1} << 2;
    } else if (states.I_L_b < 0.0) {
        mask |= uint64_t{1} << 3;
    }
    if (states.I_L_c > 0.0) {
        mask |= uint64_t{1} << 4;
    } else if (states.I_L_c < 0.0) {
        mask |= uint64_t{1} << 5;
    }
    return mask;
}

uint64_t completeTopologyMask(uint64_t external_closed_switch_mask, uint64_t solver_closed_diode_mask) {
    uint64_t switch_mask = external_closed_switch_mask;
    // The base mask is the external-closed topology. Solver closure
    // can close additional diode switches, but it must not open a diode
    // switch that is already external-closed.
    switch_mask |= ((solver_closed_diode_mask >> 0) & uint64_t{1}) << 0;
    switch_mask |= ((solver_closed_diode_mask >> 1) & uint64_t{1}) << 1;
    switch_mask |= ((solver_closed_diode_mask >> 2) & uint64_t{1}) << 2;
    switch_mask |= ((solver_closed_diode_mask >> 3) & uint64_t{1}) << 3;
    switch_mask |= ((solver_closed_diode_mask >> 4) & uint64_t{1}) << 4;
    switch_mask |= ((solver_closed_diode_mask >> 5) & uint64_t{1}) << 5;
    return switch_mask;
}

bool diodeExternalClosed(size_t diode_idx, uint64_t external_closed_switch_mask) {
    switch (diode_idx) {
        case 0: return (external_closed_switch_mask & (uint64_t{1} << 0)) != 0;
        case 1: return (external_closed_switch_mask & (uint64_t{1} << 1)) != 0;
        case 2: return (external_closed_switch_mask & (uint64_t{1} << 2)) != 0;
        case 3: return (external_closed_switch_mask & (uint64_t{1} << 3)) != 0;
        case 4: return (external_closed_switch_mask & (uint64_t{1} << 4)) != 0;
        case 5: return (external_closed_switch_mask & (uint64_t{1} << 5)) != 0;
    default:
        return false;
    }
}

double diodeCurrent(size_t diode_idx, Model_diode_bridge::Outputs const& outputs) {
    switch (diode_idx) {
        case 0: return outputs.I_R_D_n_a;
        case 1: return outputs.I_R_D_n_b;
        case 2: return outputs.I_R_D_n_c;
        case 3: return outputs.I_R_D_p_a;
        case 4: return outputs.I_R_D_p_b;
        case 5: return outputs.I_R_D_p_c;
    default:
        return 0.0;
    }
}

double diodeForwardOverdrive(size_t diode_idx,
                             Model_diode_bridge::Outputs const& outputs,
                             Model_diode_bridge::Inputs const& inputs) {
    // Positive overdrive means an open diode would be forward-biased
    // for this instantaneous solution.
    switch (diode_idx) {
        case 0: return outputs.N_dc_n - outputs._net0 - inputs.V_D_n_a;
        case 1: return outputs.N_dc_n - outputs._net1 - inputs.V_D_n_b;
        case 2: return outputs.N_dc_n - outputs._net2 - inputs.V_D_n_c;
        case 3: return outputs._net0 - outputs.N_dc_p - inputs.V_D_p_a;
        case 4: return outputs._net1 - outputs.N_dc_p - inputs.V_D_p_b;
        case 5: return outputs._net2 - outputs.N_dc_p - inputs.V_D_p_c;
    default:
        return 0.0;
    }
}

double inductorCurrentDiscontinuity(Model_diode_bridge::Outputs const& outputs,
                                    Model_diode_bridge::States const& states) {
    double discontinuity = 0.0;
    // Inductor current is continuous. The generated state vector stores
    // every inductor current, including dependent inductors, so continuity
    // can be checked without topology-specific knowledge.
    discontinuity = std::max(discontinuity, std::abs(outputs.data[0] - states.data[0]));
    discontinuity = std::max(discontinuity, std::abs(outputs.data[1] - states.data[1]));
    discontinuity = std::max(discontinuity, std::abs(outputs.data[2] - states.data[2]));
    return discontinuity;
}

Model_diode_bridge::Switches applySolverDiodeMask(Model_diode_bridge::Switches switches, uint64_t solver_closed_diode_mask) {
    // Generated diodes are represented as switches in the state-space
    // matrices. Set bits solver-close diodes and clear bits release
    // solver closure. External switch closure remains unchanged.
    switches.S_D_n_a.forceOutput((solver_closed_diode_mask & (uint64_t{1} << 0)) != 0 ? std::optional<bool>{true} : std::nullopt);
    switches.S_D_n_b.forceOutput((solver_closed_diode_mask & (uint64_t{1} << 1)) != 0 ? std::optional<bool>{true} : std::nullopt);
    switches.S_D_n_c.forceOutput((solver_closed_diode_mask & (uint64_t{1} << 2)) != 0 ? std::optional<bool>{true} : std::nullopt);
    switches.S_D_p_a.forceOutput((solver_closed_diode_mask & (uint64_t{1} << 3)) != 0 ? std::optional<bool>{true} : std::nullopt);
    switches.S_D_p_b.forceOutput((solver_closed_diode_mask & (uint64_t{1} << 4)) != 0 ? std::optional<bool>{true} : std::nullopt);
    switches.S_D_p_c.forceOutput((solver_closed_diode_mask & (uint64_t{1} << 5)) != 0 ? std::optional<bool>{true} : std::nullopt);
    return switches;
}

Model_diode_bridge::Switches releaseReverseCurrentDiodes(Model_diode_bridge::Components const& components,
                                                         Model_diode_bridge::States const& states,
                                                         Model_diode_bridge::Inputs const& inputs,
                                                         Model_diode_bridge::Switches const& switches) {
    uint64_t external_closed_switch_mask = externalClosedSwitchMask(switches);
    uint64_t solver_closed_diode_mask = solverClosedDiodeMask(switches);
    if (solver_closed_diode_mask == 0) {
        return switches;
    }

    // A controlled switch closing cannot fix an inductor-current
    // discontinuity by opening diodes; it only gives existing current
    // another path. The full continuity resolver is therefore reserved
    // for switch openings. On a closing-only transition, only release
    // solver-closed diodes carrying reverse current at t+0. Re-evaluate
    // after each release because it can make another diode reverse-biased.
    uint64_t updated_solver_closed_diode_mask = rlc2ss::releaseReverseCurrentDiodeMask(
        solver_closed_diode_mask,
        [&](uint64_t closed_diode_mask) {
            uint64_t switch_mask = completeTopologyMask(external_closed_switch_mask, closed_diode_mask);
            Model_diode_bridge::Outputs instantaneous_outputs =
                calcInstantaneousOutputs(components, states, inputs, switch_mask);
            uint64_t reverse_current_mask = 0;
            for (size_t diode_idx = 0; diode_idx < 6; ++diode_idx) {
                uint64_t diode_bit = uint64_t{1} << diode_idx;
                if ((closed_diode_mask & diode_bit) != 0 &&
                    diodeCurrent(diode_idx, instantaneous_outputs) < -rlc2ss::DIODE_CONTINUITY_TOLERANCE) {
                    reverse_current_mask |= diode_bit;
                }
            }
            return reverse_current_mask;
        });

    return updated_solver_closed_diode_mask != solver_closed_diode_mask
        ? applySolverDiodeMask(switches, updated_solver_closed_diode_mask)
        : switches;
}

Model_diode_bridge::Switches resolveDiodeContinuity(Model_diode_bridge::Components const& components,
                                                    Model_diode_bridge::States const& states,
                                                    Model_diode_bridge::Inputs const& inputs,
                                                    Model_diode_bridge::Switches const& switches,
                                                    uint64_t previous_topology_mask) {
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
        Model_diode_bridge::Outputs const& outputs) {
        double violation = 0.0;
        for (size_t diode_idx = 0; diode_idx < 6; ++diode_idx) {
            if ((solver_closed_diode_mask & (uint64_t{1} << diode_idx)) != 0) {
                violation = std::max(violation, -diodeCurrent(diode_idx, outputs));
            } else if (!diodeExternalClosed(diode_idx, external_closed_switch_mask)) {
                violation = std::max(violation, diodeForwardOverdrive(diode_idx, outputs, inputs));
            }
        }
        return violation;
    };

    double best_attempt_discontinuity = std::numeric_limits<double>::infinity();
    double best_attempt_complementarity_violation = std::numeric_limits<double>::infinity();
    // Evaluate one diode mask at t+0. The state vector is not advanced,
    // so any mismatch between inductor outputs and stored states is the
    // switching discontinuity caused by this topology.
    auto evaluate_mask = [&](uint64_t solver_closed_diode_mask) {
        uint64_t switch_mask = completeTopologyMask(external_closed_switch_mask, solver_closed_diode_mask);
        Model_diode_bridge::Outputs instantaneous_outputs =
            calcInstantaneousOutputs(components, states, inputs, switch_mask);
        double discontinuity = inductorCurrentDiscontinuity(instantaneous_outputs, states);
        double complementarity_violation = diode_complementarity_violation(solver_closed_diode_mask, instantaneous_outputs);
        best_attempt_discontinuity = std::min(best_attempt_discontinuity, discontinuity);
        best_attempt_complementarity_violation = std::min(best_attempt_complementarity_violation, complementarity_violation);
        return rlc2ss::DiodeContinuityMetrics{
            .discontinuity = discontinuity,
            .complementarity_violation = complementarity_violation,
        };
    };

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
    {
        std::scoped_lock<std::mutex> lock(cache_mutex);
        if (auto cached = diode_continuity_cache.find(cache_key); cached != diode_continuity_cache.end()) {
            cached_mask = cached->second;
        }
    }
    if (cached_mask) {
        rlc2ss::DiodeContinuityMetrics cached_metrics = evaluate_mask(*cached_mask);
        if (rlc2ss::diodeContinuityValid(cached_metrics, rlc2ss::DIODE_CONTINUITY_TOLERANCE)) {
            return applySolverDiodeMask(switches, *cached_mask);
        }
    }

    // Fall back to a complete mask search. The helper searches by
    // increasing diode-state changes from the current diode mask, so the
    // common one-diode freewheel case is found before wider combinations
    // are evaluated.
    rlc2ss::DiodeContinuitySelection selection = rlc2ss::selectDiodeContinuityMask(
        6,
        initial_solver_closed_diode_mask,
        rlc2ss::DIODE_CONTINUITY_TOLERANCE,
        evaluate_mask);

    if (selection.found) {
        {
            std::scoped_lock<std::mutex> lock(cache_mutex);
            diode_continuity_cache[cache_key] = selection.mask;
        }
        return applySolverDiodeMask(switches, selection.mask);
    }

    throw std::runtime_error(std::format(
        "Diode continuity resolver could not find a diode mask satisfying continuity and complementarity; "
        "switch combination {} initial diode mask {} best residual {} best complementarity violation {}",
        external_closed_switch_mask,
        initial_solver_closed_diode_mask,
        best_attempt_discontinuity,
        best_attempt_complementarity_violation));
}



} // namespace

std::optional<rlc2ss::ZeroCrossingEvent> Model_diode_bridge::checkZeroCrossingEvents(Model_diode_bridge::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    // Diode D_n_a
    double V_D_n_a = outputs.N_dc_n - outputs._net0;
    if (V_D_n_a > inputs.V_D_n_a && !switches.S_D_n_a) {
        double V_D_n_a_prev = prev_outputs.N_dc_n - prev_outputs._net0;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_n_a_prev, V_D_n_a),
            .event_callback = [this]() {
                switches.S_D_n_a.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_n_a < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_n_a.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_n_a, outputs.I_R_D_n_a),
            .event_callback = [this]() {
                switches.S_D_n_a.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_n_b
    double V_D_n_b = outputs.N_dc_n - outputs._net1;
    if (V_D_n_b > inputs.V_D_n_b && !switches.S_D_n_b) {
        double V_D_n_b_prev = prev_outputs.N_dc_n - prev_outputs._net1;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_n_b_prev, V_D_n_b),
            .event_callback = [this]() {
                switches.S_D_n_b.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_n_b < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_n_b.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_n_b, outputs.I_R_D_n_b),
            .event_callback = [this]() {
                switches.S_D_n_b.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_n_c
    double V_D_n_c = outputs.N_dc_n - outputs._net2;
    if (V_D_n_c > inputs.V_D_n_c && !switches.S_D_n_c) {
        double V_D_n_c_prev = prev_outputs.N_dc_n - prev_outputs._net2;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_n_c_prev, V_D_n_c),
            .event_callback = [this]() {
                switches.S_D_n_c.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_n_c < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_n_c.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_n_c, outputs.I_R_D_n_c),
            .event_callback = [this]() {
                switches.S_D_n_c.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_p_a
    double V_D_p_a = outputs._net0 - outputs.N_dc_p;
    if (V_D_p_a > inputs.V_D_p_a && !switches.S_D_p_a) {
        double V_D_p_a_prev = prev_outputs._net0 - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_p_a_prev, V_D_p_a),
            .event_callback = [this]() {
                switches.S_D_p_a.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_p_a < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_p_a.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_p_a, outputs.I_R_D_p_a),
            .event_callback = [this]() {
                switches.S_D_p_a.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_p_b
    double V_D_p_b = outputs._net1 - outputs.N_dc_p;
    if (V_D_p_b > inputs.V_D_p_b && !switches.S_D_p_b) {
        double V_D_p_b_prev = prev_outputs._net1 - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_p_b_prev, V_D_p_b),
            .event_callback = [this]() {
                switches.S_D_p_b.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_p_b < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_p_b.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_p_b, outputs.I_R_D_p_b),
            .event_callback = [this]() {
                switches.S_D_p_b.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_p_c
    double V_D_p_c = outputs._net2 - outputs.N_dc_p;
    if (V_D_p_c > inputs.V_D_p_c && !switches.S_D_p_c) {
        double V_D_p_c_prev = prev_outputs._net2 - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_p_c_prev, V_D_p_c),
            .event_callback = [this]() {
                switches.S_D_p_c.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_p_c < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_p_c.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_p_c, outputs.I_R_D_p_c),
            .event_callback = [this]() {
                switches.S_D_p_c.forceOutput(std::nullopt);
            }
        });
    }

    for (auto const& callback : m_zero_crossing_callbacks) {
        std::optional<rlc2ss::ZeroCrossingEvent> event = callback(prev_outputs, outputs);
        if (event) {
            events.push(*event);
        }
    }

    if (events.size() > 0) {
        return events.top();
    }
    return std::nullopt;
}

Model_diode_bridge::Model_diode_bridge(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
}



void Model_diode_bridge::addInductorSaturation(double* inductor, std::vector<double> currents, std::vector<double> inductances) {
    // Check that the currents are ascending and inductances are descending
    assert(currents.size() == inductances.size());
    for (int i = 1; i < currents.size(); ++i) {
        assert(currents[i] >= currents[i - 1]);
        assert(inductances[i] <= inductances[i - 1]);
    }
    int i_L_output_idx = -1;
    if (inductor == &components.L_a) {
        i_L_output_idx = 0;
    }
    if (inductor == &components.L_b) {
        i_L_output_idx = 1;
    }
    if (inductor == &components.L_c) {
        i_L_output_idx = 2;
    }
    if (i_L_output_idx == -1) {
        assert(("Invalid pointer to inductor", false));
    }

    for (int i = 1; i < currents.size(); ++i) {
        double threshold = currents[i];
        double inductance_prev = inductances[i - 1];
        double inductance = inductances[i];
        // Check +threshold and -threshold separately. Interpolating abs(current)
        // gives the wrong event time if current crosses through zero during a
        // step, e.g. -50 A -> +150 A with a 100 A threshold.
        // Increase inductance when current goes below level
        m_zero_crossing_callbacks.push_back([=](Outputs const& outputs_prev, Outputs const& outputs_new) -> std::optional<rlc2ss::ZeroCrossingEvent> {
            double i_prev = outputs_prev.data[i_L_output_idx];
            double i_new = outputs_new.data[i_L_output_idx];
            if (i_prev > threshold && i_new <= threshold) {
                return rlc2ss::ZeroCrossingEvent{
                    .time = rlc2ss::calcZeroCrossingTime(i_prev - threshold, i_new - threshold),
                    .event_callback = [inductor, inductance_prev]() {
                        *inductor = inductance_prev;
                    }};
            }
            if (i_prev < -threshold && i_new >= -threshold) {
                return rlc2ss::ZeroCrossingEvent{
                    .time = rlc2ss::calcZeroCrossingTime(i_prev + threshold, i_new + threshold),
                    .event_callback = [inductor, inductance_prev]() {
                        *inductor = inductance_prev;
                    }};
            }
            return std::nullopt;
        });
        // Decrease inductance when current goes above level
        m_zero_crossing_callbacks.push_back([=](Outputs const& outputs_prev, Outputs const& outputs_new) -> std::optional<rlc2ss::ZeroCrossingEvent> {
            double i_prev = outputs_prev.data[i_L_output_idx];
            double i_new = outputs_new.data[i_L_output_idx];
            if (i_prev < threshold && i_new >= threshold) {
                return rlc2ss::ZeroCrossingEvent{
                    .time = rlc2ss::calcZeroCrossingTime(i_prev - threshold, i_new - threshold),
                    .event_callback = [inductor, inductance]() {
                        *inductor = inductance;
                    }};
            }
            if (i_prev > -threshold && i_new <= -threshold) {
                return rlc2ss::ZeroCrossingEvent{
                    .time = rlc2ss::calcZeroCrossingTime(i_prev + threshold, i_new + threshold),
                    .event_callback = [inductor, inductance]() {
                        *inductor = inductance;
                    }};
            }
            return std::nullopt;
        });
    }
}

void Model_diode_bridge::step(double dt, Inputs const& inputs_) {
    inputs.data = inputs_.data;

    // Integrate each interval using the topology active before its delayed
    // switch event, then apply the event at the end of that interval.
    double smallest_dt = switches.smallestDelay();
    // Match OnOffDelay::step() tolerance so a floating-point rounding at the
    // timestep endpoint cannot make it apply an event that this loop misses.
    while (smallest_dt <= dt + rlc2ss::EPSILON) {
        double event_dt = std::min(smallest_dt, dt);
        stepWithZeroCrossingDetection(event_dt);
        switches.step(event_dt);
        dt -= event_dt;
        smallest_dt = switches.smallestDelay();
    }

    stepWithZeroCrossingDetection(dt);
    switches.step(dt);
}

void Model_diode_bridge::stepWithZeroCrossingDetection(double dt) {
    if constexpr (NUM_DIODES > 0) {
        uint64_t external_closed_switch_mask = externalClosedSwitchMask(switches);
        if (external_closed_switch_mask != m_last_external_closed_switch_mask) {
            bool first_continuity_step = m_last_external_closed_switch_mask == ~uint64_t{0};
            bool external_switch_opened = (m_last_external_closed_switch_mask & ~external_closed_switch_mask) != 0;

            // Opening a controlled switch can remove the only path for an inductor
            // current, so it may need a diode mask search. Closing a switch only
            // adds a path; diode turn-off remains a complementarity/zero-crossing
            // problem and does not need the expensive continuity resolver.
            if (first_continuity_step || external_switch_opened) {
                switches = resolveDiodeContinuity(components, states, inputs, switches, m_last_switch_mask);
                m_last_switch_mask = switches.all();
            } else {
                switches = releaseReverseCurrentDiodes(components, states, inputs, switches);
                m_last_switch_mask = switches.all();
            }
            m_last_external_closed_switch_mask = external_closed_switch_mask;
        }
    }

    // A zero-size step applies the new topology and refreshes algebraic
    // outputs at t+0 without integrating the state.
    if (dt < rlc2ss::MINIMUM_TIMESTEP) {
        stepModel(0.0);
        return;
    }

    if constexpr (NUM_DIODES == 0) {
        // Inductor saturation registers zero-crossing callbacks, so the fast
        // path is used only when neither diodes nor saturation need checking.
        if (m_zero_crossing_callbacks.empty()) {
            stepModel(dt);
            return;
        }
    }

    // Copy previous state and outputs if step needs to be redone
    Model_diode_bridge::States prev_state;
    Model_diode_bridge::Outputs prev_outputs;
    prev_state.data = states.data;
    prev_outputs.data = outputs.data;

    stepModel(dt);
    std::optional<rlc2ss::ZeroCrossingEvent> zc_event = checkZeroCrossingEvents(prev_outputs);
    int zc_event_count = 0;
    while (zc_event && zc_event_count < MAX_ZERO_CROSS_EVENTS) {
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
    }
}

void Model_diode_bridge::stepModel(double dt) {
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all() != _M_switches_DO_NOT_TOUCH.all() || !m_solver.initialized()) {
        assert(components.C_dc != -1);
        assert(components.L_a != -1);
        assert(components.L_b != -1);
        assert(components.L_c != -1);
        assert(components.R_D_n_a != -1);
        assert(components.R_D_n_b != -1);
        assert(components.R_D_n_c != -1);
        assert(components.R_D_p_a != -1);
        assert(components.R_D_p_b != -1);
        assert(components.R_D_p_c != -1);
        assert(components.R_a != -1);
        assert(components.R_b != -1);
        assert(components.R_c != -1);
        assert(components.R_dc != -1);
        assert(components.R_load != -1);
        _M_components_DO_NOT_TOUCH = components;
        _M_switches_DO_NOT_TOUCH = switches;
        m_ss = calcStateSpaceMatrices(components, switches.all());
        m_solver.updateSystem(m_ss.A, m_ss.B);
        m_backward_euler_pending = true;
    }

    if (dt >= rlc2ss::MINIMUM_TIMESTEP) {
        dt = std::max(dt, m_dt_resolution);
        // Solve the first nonzero step after a topology change with backward
        // Euler to reduce numerical oscillations.
        if (m_backward_euler_pending) {
            m_backward_euler_pending = false;
            if (m_dt_resolution > 0) {
                double multiple = std::round(dt / m_dt_resolution);
                states.data = m_solver.stepLinearBackwardEuler(states.data, inputs.data, multiple * m_dt_resolution);
            } else {
                states.data = m_solver.stepLinearBackwardEuler(states.data, inputs.data, dt);
            }
        } else {
            if (m_dt_resolution > 0) {
                if (m_dt_correction_mode == TimestepErrorCorrectionMode::NONE) {
                    // Solve with tustin as multiples of resolution and ignore any error
                    double multiple = std::round(dt / m_dt_resolution);
                    states.data = m_solver.stepLinearTustin(states.data, inputs.data, multiple * m_dt_resolution);
                } else if (m_dt_correction_mode == TimestepErrorCorrectionMode::ACCUMULATE) {
                    // Solve with tustin as multiples of resolution and accumulate error to correct the timestep length
                    // on later steps
                    double multiple = (dt + m_dt_error_accumulator) / m_dt_resolution;
                    m_dt_error_accumulator += dt - std::round(multiple) * m_dt_resolution;
                    states.data = m_solver.stepLinearTustin(states.data, inputs.data, std::round(multiple) * m_dt_resolution);
                }
            } else {
                states.data = m_solver.stepLinearTustin(states.data, inputs.data, dt);
            }
        }
    }

    // Update output
    outputs.data = m_ss.C * states.data + m_ss.D * inputs.data;

    // Update states from outputs to have correct values for dependent states
    states.I_L_a = outputs.I_L_a;
    states.I_L_b = outputs.I_L_b;
    states.I_L_c = outputs.I_L_c;
    states.V_C_dc = outputs.V_C_dc;
}

bool Model_diode_bridge::Components::operator==(Components const& other) const {
    return
        C_dc == other.C_dc &&
        L_a == other.L_a &&
        L_b == other.L_b &&
        L_c == other.L_c &&
        R_D_n_a == other.R_D_n_a &&
        R_D_n_b == other.R_D_n_b &&
        R_D_n_c == other.R_D_n_c &&
        R_D_p_a == other.R_D_p_a &&
        R_D_p_b == other.R_D_p_b &&
        R_D_p_c == other.R_D_p_c &&
        R_a == other.R_a &&
        R_b == other.R_b &&
        R_c == other.R_c &&
        R_dc == other.R_dc &&
        R_load == other.R_load;
}

uint64_t Model_diode_bridge::Components::hash() const {
    uint64_t seed = 0;
    rlc2ss::hash_combine(seed, C_dc);
    rlc2ss::hash_combine(seed, L_a);
    rlc2ss::hash_combine(seed, L_b);
    rlc2ss::hash_combine(seed, L_c);
    rlc2ss::hash_combine(seed, R_D_n_a);
    rlc2ss::hash_combine(seed, R_D_n_b);
    rlc2ss::hash_combine(seed, R_D_n_c);
    rlc2ss::hash_combine(seed, R_D_p_a);
    rlc2ss::hash_combine(seed, R_D_p_b);
    rlc2ss::hash_combine(seed, R_D_p_c);
    rlc2ss::hash_combine(seed, R_a);
    rlc2ss::hash_combine(seed, R_b);
    rlc2ss::hash_combine(seed, R_c);
    rlc2ss::hash_combine(seed, R_dc);
    rlc2ss::hash_combine(seed, R_load);
    return seed;
}

uint64_t Model_diode_bridge::Switches::all() const {
    return 0 |
        (uint64_t{S_D_n_a} << 0) |
        (uint64_t{S_D_n_b} << 1) |
        (uint64_t{S_D_n_c} << 2) |
        (uint64_t{S_D_p_a} << 3) |
        (uint64_t{S_D_p_b} << 4) |
        (uint64_t{S_D_p_c} << 5);
}

double Model_diode_bridge::Switches::smallestDelay() {
    return std::min({double(rlc2ss::OnOffDelay::MAX_DELAY),
                    S_D_n_a.pendingTime(),
                    S_D_n_b.pendingTime(),
                    S_D_n_c.pendingTime(),
                    S_D_p_a.pendingTime(),
                    S_D_p_b.pendingTime(),
                    S_D_p_c.pendingTime()});
}

void Model_diode_bridge::Switches::step(double dt) {
    S_D_n_a.step(dt);
    S_D_n_b.step(dt);
    S_D_n_c.step(dt);
    S_D_p_a.step(dt);
    S_D_p_b.step(dt);
    S_D_p_c.step(dt);
}
