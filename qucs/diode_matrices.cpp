
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

static std::unique_ptr<Model_diode::StateSpaceMatrices> calcStateSpace(
    Eigen::MatrixXd const& K1,
    Eigen::MatrixXd const& A1,
    Eigen::MatrixXd const& B1,
    Eigen::MatrixXd const& K2,
    Eigen::MatrixXd const& C1,
    Eigen::MatrixXd const& D1) {
    auto ss = std::make_unique<Model_diode::StateSpaceMatrices>();
    Eigen::MatrixXd A = K1.partialPivLu().solve(A1);
    Eigen::MatrixXd B = K1.partialPivLu().solve(B1);
    ss->A = A;
    ss->B = B;
    ss->C = (C1 + K2 * A);
    ss->D = (D1 + K2 * B);
    return ss;
}

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


Model_diode::Outputs Model_diode::calcInstantaneousOutputs(uint64_t switch_combination) {
    Outputs instantaneous_outputs;
    // Evaluate the algebraic outputs for an explicit switch mask at t+0.
    // The state vector is not advanced, so any mismatch between inductor
    // output currents and stored inductor states is a real switching
    // discontinuity.
    StateSpaceMatrices const& ss = calcStateSpaceMatrices(switch_combination);
    instantaneous_outputs.data = ss.C * states.data + ss.D * inputs.data;
    return instantaneous_outputs;
}

uint64_t Model_diode::controlledSwitchMask() const {
    // Track each switch's delayed control output, not its possibly
    // diode-forced actual value. This keeps diode zero-crossing
    // force/release events out of controlled topology detection,
    // while still detecting explicit control of a diode switch.
    return 0 |
        (uint64_t{switches.S1.output()} << 0) |
        (uint64_t{switches.S_D1.output()} << 1) |
        (uint64_t{switches.S_D2.output()} << 2) |
        (uint64_t{switches.S_D3.output()} << 3);
}

uint64_t Model_diode::closedDiodeMask() const {
    uint64_t mask = 0;
    for (size_t diode_idx = 0; diode_idx < 3; ++diode_idx) {
        if (diodeClosed(diode_idx)) {
            mask |= uint64_t{1} << diode_idx;
        }
    }
    return mask;
}

uint64_t Model_diode::inductorCurrentSignMask() const {
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

uint64_t Model_diode::switchMaskWithClosedDiodes(uint64_t base_switch_mask, uint64_t closed_diode_mask) const {
    uint64_t switch_mask = base_switch_mask;
    // The base mask is the controlled-switch topology. Diode forces
    // can add closed diode switches, but they must not clear a switch
    // that is closed by its controlled output.
    if ((closed_diode_mask & (uint64_t{1} << 0)) != 0) {
        switch_mask |= uint64_t{1} << 1;
    }
    if ((closed_diode_mask & (uint64_t{1} << 1)) != 0) {
        switch_mask |= uint64_t{1} << 2;
    }
    if ((closed_diode_mask & (uint64_t{1} << 2)) != 0) {
        switch_mask |= uint64_t{1} << 3;
    }
    return switch_mask;
}

bool Model_diode::diodeClosed(size_t diode_idx) const {
    // This is the diode-forced state only. A diode switch can also be
    // closed by its controlled output; that base topology is tracked
    // separately by controlledSwitchMask().
    switch (diode_idx) {
        case 0: return switches.S_D1.forcedOutput().value_or(false);
        case 1: return switches.S_D2.forcedOutput().value_or(false);
        case 2: return switches.S_D3.forcedOutput().value_or(false);
    default:
        return false;
    }
}

bool Model_diode::diodeControlledClosed(size_t diode_idx, uint64_t controlled_switch_mask) const {
    switch (diode_idx) {
        case 0: return (controlled_switch_mask & (uint64_t{1} << 1)) != 0;
        case 1: return (controlled_switch_mask & (uint64_t{1} << 2)) != 0;
        case 2: return (controlled_switch_mask & (uint64_t{1} << 3)) != 0;
    default:
        return false;
    }
}

double Model_diode::diodeCurrent(size_t diode_idx, Outputs const& outputs_) const {
    switch (diode_idx) {
        case 0: return outputs_.I_R_D1;
        case 1: return outputs_.I_R_D2;
        case 2: return outputs_.I_R_D3;
    default:
        return 0.0;
    }
}

double Model_diode::diodeForwardOverdrive(size_t diode_idx, Outputs const& outputs_) const {
    // Positive overdrive means an open diode would be forward-biased
    // for this instantaneous solution.
    switch (diode_idx) {
        case 0: return outputs_._net2 - outputs_._net1 - inputs.V_D1;
        case 1: return outputs_._net4 - outputs_.N_D2_P - inputs.V_D2;
        case 2: return outputs_.N_D3_N - outputs_._net4 - inputs.V_D3;
    default:
        return 0.0;
    }
}

double Model_diode::inductorCurrentDiscontinuity(Outputs const& outputs_) const {
    double discontinuity = 0.0;
    // Inductor current is continuous. The generated state vector stores
    // every inductor current, including dependent inductors, so continuity
    // can be checked without topology-specific knowledge.
    discontinuity = std::max(discontinuity, std::abs(outputs_.data[0] - states.data[0]));
    discontinuity = std::max(discontinuity, std::abs(outputs_.data[1] - states.data[1]));
    discontinuity = std::max(discontinuity, std::abs(outputs_.data[2] - states.data[2]));
    return discontinuity;
}

void Model_diode::forceClosedDiodeMask(uint64_t closed_diode_mask) {
    // Generated diodes are represented as switches in the state-space
    // matrices. Setting a bit here forces that diode closed;
    // clearing a bit releases the force, which leaves generated diode
    // outputs open until diode zero-crossing logic forces them closed.
    for (size_t diode_idx = 0; diode_idx < 3; ++diode_idx) {
        switch (diode_idx) {
        case 0:
            switches.S_D1.forceOutput((closed_diode_mask & (uint64_t{1} << 0)) != 0 ? std::optional<bool>{true} : std::nullopt);
            break;
        case 1:
            switches.S_D2.forceOutput((closed_diode_mask & (uint64_t{1} << 1)) != 0 ? std::optional<bool>{true} : std::nullopt);
            break;
        case 2:
            switches.S_D3.forceOutput((closed_diode_mask & (uint64_t{1} << 2)) != 0 ? std::optional<bool>{true} : std::nullopt);
            break;
        default:
            break;
        }
    }
}

void Model_diode::releaseReverseCurrentDiodes() {
    uint64_t current_switch_mask = controlledSwitchMask();
    uint64_t closed_diode_mask = closedDiodeMask();
    if (closed_diode_mask == 0) {
        m_last_switch_mask = current_switch_mask;
        return;
    }

    // A controlled switch closing cannot fix an inductor-current
    // discontinuity by opening diodes; it only gives existing current
    // another path. The full continuity resolver is therefore reserved
    // for switch openings. On a closing-only transition, only release
    // forced diodes that are already carrying reverse current at t+0.
    uint64_t switch_mask = switchMaskWithClosedDiodes(current_switch_mask, closed_diode_mask);
    Outputs instantaneous_outputs = calcInstantaneousOutputs(switch_mask);
    uint64_t updated_closed_diode_mask = closed_diode_mask;
    for (size_t diode_idx = 0; diode_idx < 3; ++diode_idx) {
        uint64_t diode_bit = uint64_t{1} << diode_idx;
        if ((closed_diode_mask & diode_bit) != 0 &&
            diodeCurrent(diode_idx, instantaneous_outputs) < -rlc2ss::DIODE_CONTINUITY_TOLERANCE) {
            updated_closed_diode_mask &= ~diode_bit;
        }
    }

    if (updated_closed_diode_mask != closed_diode_mask) {
        forceClosedDiodeMask(updated_closed_diode_mask);
    }
    m_last_switch_mask = switchMaskWithClosedDiodes(current_switch_mask, updated_closed_diode_mask);
}

void Model_diode::resolveDiodeContinuity() {
    uint64_t current_switch_mask = controlledSwitchMask();
    uint64_t initial_closed_diode_mask = closedDiodeMask();

    // Check the diode complementarity part of the candidate solution.
    // Closed diodes may conduct zero or positive current. Open diodes
    // must not be forward-biased.
    auto diode_complementarity_violation = [this, current_switch_mask](uint64_t closed_diode_mask, Outputs const& outputs_) {
        double violation = 0.0;
        for (size_t diode_idx = 0; diode_idx < 3; ++diode_idx) {
            if ((closed_diode_mask & (uint64_t{1} << diode_idx)) != 0) {
                violation = std::max(violation, -diodeCurrent(diode_idx, outputs_));
            } else if (!diodeControlledClosed(diode_idx, current_switch_mask)) {
                violation = std::max(violation, diodeForwardOverdrive(diode_idx, outputs_));
            }
        }
        return violation;
    };

    double best_attempt_discontinuity = std::numeric_limits<double>::infinity();
    double best_attempt_complementarity_violation = std::numeric_limits<double>::infinity();
    // Evaluate one diode mask at t+0. The state vector is not advanced,
    // so any mismatch between inductor outputs and stored states is the
    // switching discontinuity caused by this topology.
    auto evaluate_mask = [&](uint64_t closed_diode_mask) {
        uint64_t switch_mask = switchMaskWithClosedDiodes(current_switch_mask, closed_diode_mask);
        Outputs instantaneous_outputs = calcInstantaneousOutputs(switch_mask);
        double discontinuity = inductorCurrentDiscontinuity(instantaneous_outputs);
        double complementarity_violation = diode_complementarity_violation(closed_diode_mask, instantaneous_outputs);
        best_attempt_discontinuity = std::min(best_attempt_discontinuity, discontinuity);
        best_attempt_complementarity_violation = std::min(best_attempt_complementarity_violation, complementarity_violation);
        return rlc2ss::DiodeContinuityMetrics{
            .discontinuity = discontinuity,
            .complementarity_violation = complementarity_violation,
        };
    };

    // Cache only a warm-start set of closed diodes. The same switch
    // transition can need different diodes depending on current direction,
    // so the key includes the inductor-current sign pattern. A cached mask
    // is still fully revalidated before use.
    uint64_t cache_key = 0;
    rlc2ss::hash_combine(cache_key, m_last_switch_mask);
    rlc2ss::hash_combine(cache_key, current_switch_mask);
    rlc2ss::hash_combine(cache_key, initial_closed_diode_mask);
    rlc2ss::hash_combine(cache_key, inductorCurrentSignMask());

    if (auto cached = m_diode_continuity_cache.find(cache_key); cached != m_diode_continuity_cache.end()) {
        rlc2ss::DiodeContinuityMetrics cached_metrics = evaluate_mask(cached->second);
        if (rlc2ss::diodeContinuityValid(cached_metrics, rlc2ss::DIODE_CONTINUITY_TOLERANCE)) {
            uint64_t cached_switch_mask = switchMaskWithClosedDiodes(current_switch_mask, cached->second);
            forceClosedDiodeMask(cached->second);
            m_last_switch_mask = cached_switch_mask;
            return;
        }
    }

    // Fall back to a complete mask search. The helper searches by
    // increasing diode-state changes from the current diode mask, so the
    // common one-diode freewheel case is found before wider combinations
    // are evaluated.
    rlc2ss::DiodeContinuitySelection selection = rlc2ss::selectDiodeContinuityMask(
        3,
        initial_closed_diode_mask,
        rlc2ss::DIODE_CONTINUITY_TOLERANCE,
        evaluate_mask);

    if (selection.found) {
        uint64_t final_switch_mask = switchMaskWithClosedDiodes(current_switch_mask, selection.mask);
        forceClosedDiodeMask(selection.mask);
        m_diode_continuity_cache[cache_key] = selection.mask;
        m_last_switch_mask = final_switch_mask;
        return;
    }

    throw std::runtime_error(std::format(
        "Diode continuity resolver could not find a diode mask satisfying continuity and complementarity; "
        "switch combination {} initial diode mask {} best residual {} best complementarity violation {}",
        current_switch_mask,
        initial_closed_diode_mask,
        best_attempt_discontinuity,
        best_attempt_complementarity_violation));
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

    uint64_t continuity_switch_mask = controlledSwitchMask();
    if (continuity_switch_mask != m_last_continuity_switch_mask) {
        bool first_continuity_step = m_last_continuity_switch_mask == ~uint64_t{0};
        bool controlled_switch_opened = (m_last_continuity_switch_mask & ~continuity_switch_mask) != 0;

        // Opening a controlled switch can remove the only path for an inductor
        // current, so it may need a diode mask search. Closing a switch only
        // adds a path; diode turn-off remains a complementarity/zero-crossing
        // problem and does not need the expensive continuity resolver.
        if (first_continuity_step || controlled_switch_opened) {
            resolveDiodeContinuity();
        } else {
            releaseReverseCurrentDiodes();
        }
        m_last_continuity_switch_mask = continuity_switch_mask;
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
        m_ss = calcStateSpaceMatrices(switches.all());
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

Model_diode::StateSpaceMatrices const& Model_diode::calcStateSpaceMatrices(uint64_t switch_combination) {
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
