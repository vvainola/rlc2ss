
#include "diode_matrices.hpp"
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

std::unique_ptr<Model_diode::StateSpaceMatrices> calcStateSpace(Eigen::MatrixXd const& K1,
                                                                Eigen::MatrixXd const& A1,
                                                                Eigen::MatrixXd const& B1,
                                                                Eigen::MatrixXd const& K2,
                                                                Eigen::MatrixXd const& C1,
                                                                Eigen::MatrixXd const& D1) {
    auto ss = std::make_unique<Model_diode::StateSpaceMatrices>();
    auto lu = K1.partialPivLu();
    Eigen::MatrixXd A = lu.solve(A1);
    Eigen::MatrixXd B = lu.solve(B1);
    ss->A = A;
    ss->B = B;
    ss->C = (C1 + K2 * A);
    ss->D = (D1 + K2 * B);
    return ss;
}

Model_diode::StateSpaceMatrices const& calcStateSpaceMatrices(Model_diode::Components const& components,
                                                              uint64_t switch_combination) {
    static std::mutex            cache_mutex;
    std::scoped_lock<std::mutex> lock(cache_mutex);

    using StateSpaceMap = std::unordered_map<uint64_t, std::unique_ptr<Model_diode::StateSpaceMatrices>>;
    static std::unordered_map<uint64_t, StateSpaceMap> state_space_cache;
    uint64_t component_hash = components.hash();
    if (state_space_cache.contains(switch_combination)) {
        std::unordered_map<uint64_t, std::unique_ptr<Model_diode::StateSpaceMatrices>>& cache = state_space_cache.at(switch_combination);
        auto it = cache.find(component_hash);
        if (it != cache.end()) {
            return *it->second;
        }
    }
    std::string netlist = "L2 _net0 _net1 1E-2; \nV1 _net1 0 DC 1 \nR1 _net2 N_D2_P 0.1; tc1=0.0 tc2=0.0 \nR2 0 N_D3_N 1; tc1=0.0 tc2=0.0 \nR3 N_D2_P _net3 1; tc1=0.0 tc2=0.0 \nR4 0 _net0 1; tc1=0.0 tc2=0.0 \nD1 _net2 _net1 DMOD_D1 AREA=1.0 \nD2 _net4 N_D2_P DMOD_D2 AREA=1.0 \nD3 N_D3_N _net4 DMOD_D3 AREA=1.0 \nL3 _net5 _net6 2E-2; \nL1 _net3 N_D3_N 2E-2; \nR5 _net5 _net6 1E-3; tc1=0.0 tc2=0.0 \nK1 L1 L3 0.5; \nS1 _net1 _net2 _net7 _net8 ";

    // Cache symbolic intermediate matrices per switch combination
    static std::unordered_map<uint64_t, rlc2ss::SymbolicStateSpace> symbolic_cache;
    if (!symbolic_cache.contains(switch_combination)) {
        symbolic_cache[switch_combination] = rlc2ss::formStateSpaceMatrices(netlist, switch_combination);
    }
    rlc2ss::SymbolicStateSpace const& symbolic_ss = symbolic_cache[switch_combination];

    // Substitute component values into cached symbolic matrices
    std::unordered_map<std::string, double> values{
        {"K1", components.K1},
        {"L1", components.L1},
        {"L2", components.L2},
        {"L3", components.L3},
        {"R1", components.R1},
        {"R2", components.R2},
        {"R3", components.R3},
        {"R4", components.R4},
        {"R5", components.R5},
        {"R_D1", components.R_D1},
        {"R_D2", components.R_D2},
        {"R_D3", components.R_D3},
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


Model_diode::Outputs calcInstantaneousOutputs(Model_diode::Components const& components,
                                              Model_diode::States const& states,
                                              Model_diode::Inputs const& inputs,
                                              uint64_t switch_combination) {
    Model_diode::Outputs instantaneous_outputs;
    // Evaluate the algebraic outputs for an explicit switch mask at t+0.
    // The state vector is not advanced, so any mismatch between inductor
    // output currents and stored inductor states is a real switching
    // discontinuity.
    auto const& ss = calcStateSpaceMatrices(components, switch_combination);
    instantaneous_outputs.data = ss.C * states.data + ss.D * inputs.data;
    return instantaneous_outputs;
}

uint64_t externalClosedSwitchMask(Model_diode::Switches const& switches) {
    // Track each switch's delayed control output, not its possibly
    // solver-closed actual value. This keeps diode zero-crossing
    // force/release events out of external-closure detection while still
    // detecting external closure of a diode switch.
    return 0 |
        (uint64_t{switches.S1.output()} << 0) |
        (uint64_t{switches.S_D1.output()} << 1) |
        (uint64_t{switches.S_D2.output()} << 2) |
        (uint64_t{switches.S_D3.output()} << 3);
}

bool diodeSolverClosed(Model_diode::Switches const& switches, size_t diode_idx) {
    // True when diode continuity or zero-crossing logic has closed the
    // diode. This excludes closure caused by the external switch control,
    // which is tracked separately by externalClosedSwitchMask().
    switch (diode_idx) {
        case 0: return switches.S_D1.forcedOutput().value_or(false);
        case 1: return switches.S_D2.forcedOutput().value_or(false);
        case 2: return switches.S_D3.forcedOutput().value_or(false);
    default:
        return false;
    }
}

uint64_t solverClosedDiodeMask(Model_diode::Switches const& switches) {
    uint64_t mask = 0;
    for (size_t diode_idx = 0; diode_idx < 3; ++diode_idx) {
        if (diodeSolverClosed(switches, diode_idx)) {
            mask |= uint64_t{1} << diode_idx;
        }
    }
    return mask;
}

uint64_t inductorCurrentSignMask(Model_diode::States const& states) {
    uint64_t mask = 0;
    if (states.I_L1 > 0.0) {
        mask |= uint64_t{1} << 0;
    } else if (states.I_L1 < 0.0) {
        mask |= uint64_t{1} << 1;
    }
    if (states.I_L2 > 0.0) {
        mask |= uint64_t{1} << 2;
    } else if (states.I_L2 < 0.0) {
        mask |= uint64_t{1} << 3;
    }
    if (states.I_L3 > 0.0) {
        mask |= uint64_t{1} << 4;
    } else if (states.I_L3 < 0.0) {
        mask |= uint64_t{1} << 5;
    }
    return mask;
}

uint64_t completeTopologyMask(uint64_t external_closed_switch_mask, uint64_t solver_closed_diode_mask) {
    uint64_t switch_mask = external_closed_switch_mask;
    // The base mask is the external-closed topology. Solver closure
    // can close additional diode switches, but it must not open a diode
    // switch that is already external-closed.
    switch_mask |= ((solver_closed_diode_mask >> 0) & uint64_t{1}) << 1;
    switch_mask |= ((solver_closed_diode_mask >> 1) & uint64_t{1}) << 2;
    switch_mask |= ((solver_closed_diode_mask >> 2) & uint64_t{1}) << 3;
    return switch_mask;
}

bool diodeExternalClosed(size_t diode_idx, uint64_t external_closed_switch_mask) {
    switch (diode_idx) {
        case 0: return (external_closed_switch_mask & (uint64_t{1} << 1)) != 0;
        case 1: return (external_closed_switch_mask & (uint64_t{1} << 2)) != 0;
        case 2: return (external_closed_switch_mask & (uint64_t{1} << 3)) != 0;
    default:
        return false;
    }
}

double diodeCurrent(size_t diode_idx, Model_diode::Outputs const& outputs) {
    switch (diode_idx) {
        case 0: return outputs.I_R_D1;
        case 1: return outputs.I_R_D2;
        case 2: return outputs.I_R_D3;
    default:
        return 0.0;
    }
}

double diodeForwardOverdrive(size_t diode_idx,
                             Model_diode::Outputs const& outputs,
                             Model_diode::Inputs const& inputs) {
    // Positive overdrive means an open diode would be forward-biased
    // for this instantaneous solution.
    switch (diode_idx) {
        case 0: return outputs._net2 - outputs._net1 - inputs.V_D1;
        case 1: return outputs._net4 - outputs.N_D2_P - inputs.V_D2;
        case 2: return outputs.N_D3_N - outputs._net4 - inputs.V_D3;
    default:
        return 0.0;
    }
}

double inductorCurrentDiscontinuity(Model_diode::Outputs const& outputs,
                                    Model_diode::States const& states) {
    double discontinuity = 0.0;
    // Inductor current is continuous. The generated state vector stores
    // every inductor current, including dependent inductors, so continuity
    // can be checked without topology-specific knowledge.
    discontinuity = std::max(discontinuity, std::abs(outputs.data[0] - states.data[0]));
    discontinuity = std::max(discontinuity, std::abs(outputs.data[1] - states.data[1]));
    discontinuity = std::max(discontinuity, std::abs(outputs.data[2] - states.data[2]));
    return discontinuity;
}

Model_diode::Switches applySolverDiodeMask(Model_diode::Switches switches, uint64_t solver_closed_diode_mask) {
    // Generated diodes are represented as switches in the state-space
    // matrices. Set bits solver-close diodes and clear bits release
    // solver closure. External switch closure remains unchanged.
    switches.S_D1.forceOutput((solver_closed_diode_mask & (uint64_t{1} << 0)) != 0 ? std::optional<bool>{true} : std::nullopt);
    switches.S_D2.forceOutput((solver_closed_diode_mask & (uint64_t{1} << 1)) != 0 ? std::optional<bool>{true} : std::nullopt);
    switches.S_D3.forceOutput((solver_closed_diode_mask & (uint64_t{1} << 2)) != 0 ? std::optional<bool>{true} : std::nullopt);
    return switches;
}

Model_diode::Switches releaseReverseCurrentDiodes(Model_diode::Components const& components,
                                                  Model_diode::States const& states,
                                                  Model_diode::Inputs const& inputs,
                                                  Model_diode::Switches const& switches) {
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
            Model_diode::Outputs instantaneous_outputs =
                calcInstantaneousOutputs(components, states, inputs, switch_mask);
            uint64_t reverse_current_mask = 0;
            for (size_t diode_idx = 0; diode_idx < 3; ++diode_idx) {
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

Model_diode::Switches resolveDiodeContinuity(Model_diode::Components const& components,
                                             Model_diode::States const& states,
                                             Model_diode::Inputs const& inputs,
                                             Model_diode::Switches const& switches,
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
        Model_diode::Outputs const& outputs) {
        double violation = 0.0;
        for (size_t diode_idx = 0; diode_idx < 3; ++diode_idx) {
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
        Model_diode::Outputs instantaneous_outputs =
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
        3,
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

std::optional<rlc2ss::ZeroCrossingEvent> Model_diode::checkZeroCrossingEvents(Model_diode::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    // Diode D1
    double V_D1 = outputs._net2 - outputs._net1;
    if (V_D1 > inputs.V_D1 && !switches.S_D1) {
        double V_D1_prev = prev_outputs._net2 - prev_outputs._net1;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D1_prev, V_D1),
            .event_callback = [this]() {
                switches.S_D1.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D1 < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D1.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D1, outputs.I_R_D1),
            .event_callback = [this]() {
                switches.S_D1.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D2
    double V_D2 = outputs._net4 - outputs.N_D2_P;
    if (V_D2 > inputs.V_D2 && !switches.S_D2) {
        double V_D2_prev = prev_outputs._net4 - prev_outputs.N_D2_P;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D2_prev, V_D2),
            .event_callback = [this]() {
                switches.S_D2.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D2 < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D2.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D2, outputs.I_R_D2),
            .event_callback = [this]() {
                switches.S_D2.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D3
    double V_D3 = outputs.N_D3_N - outputs._net4;
    if (V_D3 > inputs.V_D3 && !switches.S_D3) {
        double V_D3_prev = prev_outputs.N_D3_N - prev_outputs._net4;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D3_prev, V_D3),
            .event_callback = [this]() {
                switches.S_D3.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D3 < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D3.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D3, outputs.I_R_D3),
            .event_callback = [this]() {
                switches.S_D3.forceOutput(std::nullopt);
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

Model_diode::Model_diode(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
}



void Model_diode::addInductorSaturation(double* inductor, std::vector<double> currents, std::vector<double> inductances) {
    // Check that the currents are ascending and inductances are descending
    assert(currents.size() == inductances.size());
    for (int i = 1; i < currents.size(); ++i) {
        assert(currents[i] >= currents[i - 1]);
        assert(inductances[i] <= inductances[i - 1]);
    }
    int i_L_output_idx = -1;
    if (inductor == &components.L1) {
        i_L_output_idx = 0;
    }
    if (inductor == &components.L2) {
        i_L_output_idx = 1;
    }
    if (inductor == &components.L3) {
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

void Model_diode::step(double dt, Inputs const& inputs_) {
    inputs.data = inputs_.data;

    // Step to the next switching event
    double smallest_dt = switches.smallestDelay();
    while (smallest_dt < dt) {
        switches.step(smallest_dt);
        stepWithZeroCrossingDetection(smallest_dt);
        dt -= smallest_dt;
        smallest_dt = switches.smallestDelay();
    }

    // Step remaining time
    switches.step(dt);
    stepWithZeroCrossingDetection(dt);
}

void Model_diode::stepWithZeroCrossingDetection(double dt) {
    // No need to do anything
    if (dt < rlc2ss::MINIMUM_TIMESTEP) {
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

    // Copy previous state and outputs if step needs to be redone
    Model_diode::States prev_state;
    Model_diode::Outputs prev_outputs;
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

void Model_diode::stepModel(double dt) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all() != _M_switches_DO_NOT_TOUCH.all() || !m_solver.initialized()) {
        assert(components.K1 != -1);
        assert(components.L1 != -1);
        assert(components.L2 != -1);
        assert(components.L3 != -1);
        assert(components.R1 != -1);
        assert(components.R2 != -1);
        assert(components.R3 != -1);
        assert(components.R4 != -1);
        assert(components.R5 != -1);
        assert(components.R_D1 != -1);
        assert(components.R_D2 != -1);
        assert(components.R_D3 != -1);
        _M_components_DO_NOT_TOUCH = components;
        _M_switches_DO_NOT_TOUCH = switches;
        m_ss = calcStateSpaceMatrices(components, switches.all());
        m_solver.updateSystem(m_ss.A, m_ss.B);
        // Solve one step with backward euler to reduce numerical oscillations
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

    // Update output
    outputs.data = m_ss.C * states.data + m_ss.D * inputs.data;

    // Update states from outputs to have correct values for dependent states
    states.I_L1 = outputs.I_L1;
    states.I_L2 = outputs.I_L2;
    states.I_L3 = outputs.I_L3;
}

bool Model_diode::Components::operator==(Components const& other) const {
    return
        K1 == other.K1 &&
        L1 == other.L1 &&
        L2 == other.L2 &&
        L3 == other.L3 &&
        R1 == other.R1 &&
        R2 == other.R2 &&
        R3 == other.R3 &&
        R4 == other.R4 &&
        R5 == other.R5 &&
        R_D1 == other.R_D1 &&
        R_D2 == other.R_D2 &&
        R_D3 == other.R_D3;
}

uint64_t Model_diode::Components::hash() const {
    uint64_t seed = 0;
    rlc2ss::hash_combine(seed, K1);
    rlc2ss::hash_combine(seed, L1);
    rlc2ss::hash_combine(seed, L2);
    rlc2ss::hash_combine(seed, L3);
    rlc2ss::hash_combine(seed, R1);
    rlc2ss::hash_combine(seed, R2);
    rlc2ss::hash_combine(seed, R3);
    rlc2ss::hash_combine(seed, R4);
    rlc2ss::hash_combine(seed, R5);
    rlc2ss::hash_combine(seed, R_D1);
    rlc2ss::hash_combine(seed, R_D2);
    rlc2ss::hash_combine(seed, R_D3);
    return seed;
}

uint64_t Model_diode::Switches::all() const {
    return 0 |
        (uint64_t{S1} << 0) |
        (uint64_t{S_D1} << 1) |
        (uint64_t{S_D2} << 2) |
        (uint64_t{S_D3} << 3);
}

double Model_diode::Switches::smallestDelay() {
    return std::min({double(rlc2ss::OnOffDelay::MAX_DELAY),
                    S1.pendingTime(),
                    S_D1.pendingTime(),
                    S_D2.pendingTime(),
                    S_D3.pendingTime()});
}

void Model_diode::Switches::step(double dt) {
    S1.step(dt);
    S_D1.step(dt);
    S_D2.step(dt);
    S_D3.step(dt);
}
