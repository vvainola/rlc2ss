
#include "converter_matrices.hpp"
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

static std::unique_ptr<Model_converter::StateSpaceMatrices> calcStateSpace(
    Eigen::MatrixXd const& K1,
    Eigen::MatrixXd const& A1,
    Eigen::MatrixXd const& B1,
    Eigen::MatrixXd const& K2,
    Eigen::MatrixXd const& C1,
    Eigen::MatrixXd const& D1) {
    auto ss = std::make_unique<Model_converter::StateSpaceMatrices>();
    Eigen::MatrixXd A = K1.partialPivLu().solve(A1);
    Eigen::MatrixXd B = K1.partialPivLu().solve(B1);
    ss->A = A;
    ss->B = B;
    ss->C = (C1 + K2 * A);
    ss->D = (D1 + K2 * B);
    return ss;
}

std::optional<rlc2ss::ZeroCrossingEvent> Model_converter::checkZeroCrossingEvents(Model_converter::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    // Diode D_a_n
    double V_D_a_n = outputs.N_dc_n - outputs.N_c_a;
    if (V_D_a_n > inputs.V_D_a_n && !switches.S_D_a_n) {
        double V_D_a_n_prev = prev_outputs.N_dc_n - prev_outputs.N_c_a;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_a_n_prev, V_D_a_n),
            .event_callback = [this]() {
                switches.S_D_a_n.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_a_n < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_a_n.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_a_n, outputs.I_R_D_a_n),
            .event_callback = [this]() {
                switches.S_D_a_n.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_a_p
    double V_D_a_p = outputs.N_c_a - outputs.N_dc_p;
    if (V_D_a_p > inputs.V_D_a_p && !switches.S_D_a_p) {
        double V_D_a_p_prev = prev_outputs.N_c_a - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_a_p_prev, V_D_a_p),
            .event_callback = [this]() {
                switches.S_D_a_p.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_a_p < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_a_p.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_a_p, outputs.I_R_D_a_p),
            .event_callback = [this]() {
                switches.S_D_a_p.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_b_n
    double V_D_b_n = outputs.N_dc_n - outputs.N_c_b;
    if (V_D_b_n > inputs.V_D_b_n && !switches.S_D_b_n) {
        double V_D_b_n_prev = prev_outputs.N_dc_n - prev_outputs.N_c_b;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_b_n_prev, V_D_b_n),
            .event_callback = [this]() {
                switches.S_D_b_n.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_b_n < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_b_n.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_b_n, outputs.I_R_D_b_n),
            .event_callback = [this]() {
                switches.S_D_b_n.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_b_p
    double V_D_b_p = outputs.N_c_b - outputs.N_dc_p;
    if (V_D_b_p > inputs.V_D_b_p && !switches.S_D_b_p) {
        double V_D_b_p_prev = prev_outputs.N_c_b - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_b_p_prev, V_D_b_p),
            .event_callback = [this]() {
                switches.S_D_b_p.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_b_p < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_b_p.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_b_p, outputs.I_R_D_b_p),
            .event_callback = [this]() {
                switches.S_D_b_p.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_c_n
    double V_D_c_n = outputs.N_dc_n - outputs.N_c_c;
    if (V_D_c_n > inputs.V_D_c_n && !switches.S_D_c_n) {
        double V_D_c_n_prev = prev_outputs.N_dc_n - prev_outputs.N_c_c;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_c_n_prev, V_D_c_n),
            .event_callback = [this]() {
                switches.S_D_c_n.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_c_n < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_c_n.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_c_n, outputs.I_R_D_c_n),
            .event_callback = [this]() {
                switches.S_D_c_n.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_c_p
    double V_D_c_p = outputs.N_c_c - outputs.N_dc_p;
    if (V_D_c_p > inputs.V_D_c_p && !switches.S_D_c_p) {
        double V_D_c_p_prev = prev_outputs.N_c_c - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_c_p_prev, V_D_c_p),
            .event_callback = [this]() {
                switches.S_D_c_p.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_c_p < -rlc2ss::DIODE_CONTINUITY_TOLERANCE && switches.S_D_c_p.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_c_p, outputs.I_R_D_c_p),
            .event_callback = [this]() {
                switches.S_D_c_p.forceOutput(std::nullopt);
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

Model_converter::Model_converter(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
}


Model_converter::Outputs Model_converter::calcInstantaneousOutputs(uint64_t switch_combination) {
    Outputs instantaneous_outputs;
    // Evaluate the algebraic outputs for an explicit switch mask at t+0.
    // The state vector is not advanced, so any mismatch between inductor
    // output currents and stored inductor states is a real switching
    // discontinuity.
    StateSpaceMatrices const& ss = calcStateSpaceMatrices(switch_combination);
    instantaneous_outputs.data = ss.C * states.data + ss.D * inputs.data;
    return instantaneous_outputs;
}

uint64_t Model_converter::controlledSwitchMask() const {
    // Track each switch's delayed control output, not its possibly
    // diode-forced actual value. This keeps diode zero-crossing
    // force/release events out of controlled topology detection,
    // while still detecting explicit control of a diode switch.
    return 0 |
        (uint64_t{switches.S_D_a_n.output()} << 0) |
        (uint64_t{switches.S_D_a_p.output()} << 1) |
        (uint64_t{switches.S_D_b_n.output()} << 2) |
        (uint64_t{switches.S_D_b_p.output()} << 3) |
        (uint64_t{switches.S_D_c_n.output()} << 4) |
        (uint64_t{switches.S_D_c_p.output()} << 5);
}

uint64_t Model_converter::closedDiodeMask() const {
    uint64_t mask = 0;
    for (size_t diode_idx = 0; diode_idx < 6; ++diode_idx) {
        if (diodeClosed(diode_idx)) {
            mask |= uint64_t{1} << diode_idx;
        }
    }
    return mask;
}

uint64_t Model_converter::inductorCurrentSignMask() const {
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
    if (states.I_L_g_a > 0.0) {
        mask |= uint64_t{1} << 6;
    } else if (states.I_L_g_a < 0.0) {
        mask |= uint64_t{1} << 7;
    }
    if (states.I_L_g_b > 0.0) {
        mask |= uint64_t{1} << 8;
    } else if (states.I_L_g_b < 0.0) {
        mask |= uint64_t{1} << 9;
    }
    if (states.I_L_g_c > 0.0) {
        mask |= uint64_t{1} << 10;
    } else if (states.I_L_g_c < 0.0) {
        mask |= uint64_t{1} << 11;
    }
    return mask;
}

uint64_t Model_converter::switchMaskWithClosedDiodes(uint64_t base_switch_mask, uint64_t closed_diode_mask) const {
    uint64_t switch_mask = base_switch_mask;
    // The base mask is the controlled-switch topology. Diode forces
    // can add closed diode switches, but they must not clear a switch
    // that is closed by its controlled output.
    if ((closed_diode_mask & (uint64_t{1} << 0)) != 0) {
        switch_mask |= uint64_t{1} << 0;
    }
    if ((closed_diode_mask & (uint64_t{1} << 1)) != 0) {
        switch_mask |= uint64_t{1} << 1;
    }
    if ((closed_diode_mask & (uint64_t{1} << 2)) != 0) {
        switch_mask |= uint64_t{1} << 2;
    }
    if ((closed_diode_mask & (uint64_t{1} << 3)) != 0) {
        switch_mask |= uint64_t{1} << 3;
    }
    if ((closed_diode_mask & (uint64_t{1} << 4)) != 0) {
        switch_mask |= uint64_t{1} << 4;
    }
    if ((closed_diode_mask & (uint64_t{1} << 5)) != 0) {
        switch_mask |= uint64_t{1} << 5;
    }
    return switch_mask;
}

bool Model_converter::diodeClosed(size_t diode_idx) const {
    // This is the diode-forced state only. A diode switch can also be
    // closed by its controlled output; that base topology is tracked
    // separately by controlledSwitchMask().
    switch (diode_idx) {
        case 0: return switches.S_D_a_n.forcedOutput().value_or(false);
        case 1: return switches.S_D_a_p.forcedOutput().value_or(false);
        case 2: return switches.S_D_b_n.forcedOutput().value_or(false);
        case 3: return switches.S_D_b_p.forcedOutput().value_or(false);
        case 4: return switches.S_D_c_n.forcedOutput().value_or(false);
        case 5: return switches.S_D_c_p.forcedOutput().value_or(false);
    default:
        return false;
    }
}

bool Model_converter::diodeControlledClosed(size_t diode_idx, uint64_t controlled_switch_mask) const {
    switch (diode_idx) {
        case 0: return (controlled_switch_mask & (uint64_t{1} << 0)) != 0;
        case 1: return (controlled_switch_mask & (uint64_t{1} << 1)) != 0;
        case 2: return (controlled_switch_mask & (uint64_t{1} << 2)) != 0;
        case 3: return (controlled_switch_mask & (uint64_t{1} << 3)) != 0;
        case 4: return (controlled_switch_mask & (uint64_t{1} << 4)) != 0;
        case 5: return (controlled_switch_mask & (uint64_t{1} << 5)) != 0;
    default:
        return false;
    }
}

double Model_converter::diodeCurrent(size_t diode_idx, Outputs const& outputs_) const {
    switch (diode_idx) {
        case 0: return outputs_.I_R_D_a_n;
        case 1: return outputs_.I_R_D_a_p;
        case 2: return outputs_.I_R_D_b_n;
        case 3: return outputs_.I_R_D_b_p;
        case 4: return outputs_.I_R_D_c_n;
        case 5: return outputs_.I_R_D_c_p;
    default:
        return 0.0;
    }
}

double Model_converter::diodeForwardOverdrive(size_t diode_idx, Outputs const& outputs_) const {
    // Positive overdrive means an open diode would be forward-biased
    // for this instantaneous solution.
    switch (diode_idx) {
        case 0: return outputs_.N_dc_n - outputs_.N_c_a - inputs.V_D_a_n;
        case 1: return outputs_.N_c_a - outputs_.N_dc_p - inputs.V_D_a_p;
        case 2: return outputs_.N_dc_n - outputs_.N_c_b - inputs.V_D_b_n;
        case 3: return outputs_.N_c_b - outputs_.N_dc_p - inputs.V_D_b_p;
        case 4: return outputs_.N_dc_n - outputs_.N_c_c - inputs.V_D_c_n;
        case 5: return outputs_.N_c_c - outputs_.N_dc_p - inputs.V_D_c_p;
    default:
        return 0.0;
    }
}

double Model_converter::inductorCurrentDiscontinuity(Outputs const& outputs_) const {
    double discontinuity = 0.0;
    // Inductor current is continuous. The generated state vector stores
    // every inductor current, including dependent inductors, so continuity
    // can be checked without topology-specific knowledge.
    discontinuity = std::max(discontinuity, std::abs(outputs_.data[5] - states.data[0]));
    discontinuity = std::max(discontinuity, std::abs(outputs_.data[6] - states.data[1]));
    discontinuity = std::max(discontinuity, std::abs(outputs_.data[7] - states.data[2]));
    discontinuity = std::max(discontinuity, std::abs(outputs_.data[8] - states.data[3]));
    discontinuity = std::max(discontinuity, std::abs(outputs_.data[9] - states.data[4]));
    discontinuity = std::max(discontinuity, std::abs(outputs_.data[10] - states.data[5]));
    return discontinuity;
}

void Model_converter::forceClosedDiodeMask(uint64_t closed_diode_mask) {
    // Generated diodes are represented as switches in the state-space
    // matrices. Setting a bit here forces that diode closed;
    // clearing a bit releases the force, which leaves generated diode
    // outputs open until diode zero-crossing logic forces them closed.
    for (size_t diode_idx = 0; diode_idx < 6; ++diode_idx) {
        switch (diode_idx) {
        case 0:
            switches.S_D_a_n.forceOutput((closed_diode_mask & (uint64_t{1} << 0)) != 0 ? std::optional<bool>{true} : std::nullopt);
            break;
        case 1:
            switches.S_D_a_p.forceOutput((closed_diode_mask & (uint64_t{1} << 1)) != 0 ? std::optional<bool>{true} : std::nullopt);
            break;
        case 2:
            switches.S_D_b_n.forceOutput((closed_diode_mask & (uint64_t{1} << 2)) != 0 ? std::optional<bool>{true} : std::nullopt);
            break;
        case 3:
            switches.S_D_b_p.forceOutput((closed_diode_mask & (uint64_t{1} << 3)) != 0 ? std::optional<bool>{true} : std::nullopt);
            break;
        case 4:
            switches.S_D_c_n.forceOutput((closed_diode_mask & (uint64_t{1} << 4)) != 0 ? std::optional<bool>{true} : std::nullopt);
            break;
        case 5:
            switches.S_D_c_p.forceOutput((closed_diode_mask & (uint64_t{1} << 5)) != 0 ? std::optional<bool>{true} : std::nullopt);
            break;
        default:
            break;
        }
    }
}

void Model_converter::releaseReverseCurrentDiodes() {
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
    for (size_t diode_idx = 0; diode_idx < 6; ++diode_idx) {
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

void Model_converter::resolveDiodeContinuity() {
    uint64_t current_switch_mask = controlledSwitchMask();
    uint64_t initial_closed_diode_mask = closedDiodeMask();

    // Check the diode complementarity part of the candidate solution.
    // Closed diodes may conduct zero or positive current. Open diodes
    // must not be forward-biased.
    auto diode_complementarity_violation = [this, current_switch_mask](uint64_t closed_diode_mask, Outputs const& outputs_) {
        double violation = 0.0;
        for (size_t diode_idx = 0; diode_idx < 6; ++diode_idx) {
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
        6,
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


void Model_converter::addInductorSaturation(double* inductor, std::vector<double> currents, std::vector<double> inductances) {
    // Check that the currents are ascending and inductances are descending
    assert(currents.size() == inductances.size());
    for (int i = 1; i < currents.size(); ++i) {
        assert(currents[i] >= currents[i - 1]);
        assert(inductances[i] <= inductances[i - 1]);
    }
    int i_L_output_idx = -1;
    if (inductor == &components.L_a) {
        i_L_output_idx = 5;
    }
    if (inductor == &components.L_b) {
        i_L_output_idx = 6;
    }
    if (inductor == &components.L_c) {
        i_L_output_idx = 7;
    }
    if (inductor == &components.L_g_a) {
        i_L_output_idx = 8;
    }
    if (inductor == &components.L_g_b) {
        i_L_output_idx = 9;
    }
    if (inductor == &components.L_g_c) {
        i_L_output_idx = 10;
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

void Model_converter::step(double dt, Inputs const& inputs_) {
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

void Model_converter::stepWithZeroCrossingDetection(double dt) {
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
    Model_converter::States prev_state;
    Model_converter::Outputs prev_outputs;
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

void Model_converter::stepModel(double dt) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all() != _M_switches_DO_NOT_TOUCH.all() || !m_solver.initialized()) {
        assert(components.C_a != -1);
        assert(components.C_b != -1);
        assert(components.C_c != -1);
        assert(components.C_n != -1);
        assert(components.C_p != -1);
        assert(components.L_a != -1);
        assert(components.L_b != -1);
        assert(components.L_c != -1);
        assert(components.L_g_a != -1);
        assert(components.L_g_b != -1);
        assert(components.L_g_c != -1);
        assert(components.R_D_a_n != -1);
        assert(components.R_D_a_p != -1);
        assert(components.R_D_b_n != -1);
        assert(components.R_D_b_p != -1);
        assert(components.R_D_c_n != -1);
        assert(components.R_D_c_p != -1);
        assert(components.R_a != -1);
        assert(components.R_b != -1);
        assert(components.R_c != -1);
        assert(components.R_dc != -1);
        assert(components.R_g_a != -1);
        assert(components.R_g_b != -1);
        assert(components.R_g_c != -1);
        assert(components.R_n_p != -1);
        assert(components.R_n_s != -1);
        assert(components.R_p_p != -1);
        assert(components.R_p_s != -1);
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
    states.I_L_a = outputs.I_L_a;
    states.I_L_b = outputs.I_L_b;
    states.I_L_c = outputs.I_L_c;
    states.I_L_g_a = outputs.I_L_g_a;
    states.I_L_g_b = outputs.I_L_g_b;
    states.I_L_g_c = outputs.I_L_g_c;
    states.V_C_a = outputs.V_C_a;
    states.V_C_b = outputs.V_C_b;
    states.V_C_c = outputs.V_C_c;
    states.V_C_n = outputs.V_C_n;
    states.V_C_p = outputs.V_C_p;
}

Model_converter::StateSpaceMatrices const& Model_converter::calcStateSpaceMatrices(uint64_t switch_combination) {
    static std::mutex            cache_mutex;
    std::scoped_lock<std::mutex> lock(cache_mutex);

    using StateSpaceMap = std::unordered_map<uint64_t, std::unique_ptr<Model_converter::StateSpaceMatrices>>;
    static std::unordered_map<uint64_t, StateSpaceMap> state_space_cache;
    uint64_t component_hash = components.hash();
    if (state_space_cache.contains(switch_combination)) {
        std::unordered_map<uint64_t, std::unique_ptr<Model_converter::StateSpaceMatrices>>& cache = state_space_cache.at(switch_combination);
        auto it = cache.find(component_hash);
        if (it != cache.end()) {
            return *it->second;
        }
    }
    std::string netlist = "R_p_p 0 N_dc_p 1E3 \nR_n_p N_dc_n 0 1E3 \nV_dc _net0 N_dc_n DC 1 \nL_b _net1 _net2 1M \nL_a _net3 _net4 1M \nL_c _net5 _net6 1M \nC_p 0 _net7 10E-3;I; \nC_n _net8 0 10E-3;I; \nR_n_s N_dc_n _net8 10E-3 \nR_p_s _net7 N_dc_p 10E-3 \nR_a N_c_a _net3 10E-3 \nR_b N_c_b _net1 10E-3 \nR_c N_c_c _net5 10E-3 \nR_dc _net0 N_dc_p 1;I; \nV_a _net9 _net10 DC 0 SIN(0 1 1K 0 0 0) AC 1 ACPHASE 0 \nV_c _net11 _net10 DC 0 SIN(0 1 1K 0 0 0) AC 1 ACPHASE 0 \nV_b _net12 _net10 DC 0 SIN(0 1 1K 0 0 0) AC 1 ACPHASE 0 \nL_g_b _net13 _net12 1M \nL_g_a _net14 _net9 1M \nL_g_c _net15 _net11 1M \nR_g_a _net4 _net14 10E-3 \nR_g_b _net2 _net13 10E-3 \nR_g_c _net6 _net15 10E-3 \nC_a _net16 _net4 10E-3;I; \nC_b _net16 _net2 10E-3;I; \nC_c _net16 _net6 10E-3;I; \nD_a_n N_dc_n N_c_a \nD_a_p N_c_a N_dc_p \nD_b_p N_c_b N_dc_p \nD_c_p N_c_c N_dc_p \nD_c_n N_dc_n N_c_c \nD_b_n N_dc_n N_c_b ";

    // Cache symbolic intermediate matrices per switch combination
    static std::unordered_map<uint64_t, rlc2ss::SymbolicStateSpace> symbolic_cache;
    if (!symbolic_cache.contains(switch_combination)) {
        symbolic_cache[switch_combination] = rlc2ss::formStateSpaceMatrices(netlist, switch_combination);
    }
    rlc2ss::SymbolicStateSpace const& symbolic_ss = symbolic_cache[switch_combination];

    // Substitute component values into cached symbolic matrices
    std::unordered_map<std::string, double> values{
        {"C_a", components.C_a},
        {"C_b", components.C_b},
        {"C_c", components.C_c},
        {"C_n", components.C_n},
        {"C_p", components.C_p},
        {"L_a", components.L_a},
        {"L_b", components.L_b},
        {"L_c", components.L_c},
        {"L_g_a", components.L_g_a},
        {"L_g_b", components.L_g_b},
        {"L_g_c", components.L_g_c},
        {"R_D_a_n", components.R_D_a_n},
        {"R_D_a_p", components.R_D_a_p},
        {"R_D_b_n", components.R_D_b_n},
        {"R_D_b_p", components.R_D_b_p},
        {"R_D_c_n", components.R_D_c_n},
        {"R_D_c_p", components.R_D_c_p},
        {"R_a", components.R_a},
        {"R_b", components.R_b},
        {"R_c", components.R_c},
        {"R_dc", components.R_dc},
        {"R_g_a", components.R_g_a},
        {"R_g_b", components.R_g_b},
        {"R_g_c", components.R_g_c},
        {"R_n_p", components.R_n_p},
        {"R_n_s", components.R_n_s},
        {"R_p_p", components.R_p_p},
        {"R_p_s", components.R_p_s},
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

bool Model_converter::Components::operator==(Components const& other) const {
    return
        C_a == other.C_a &&
        C_b == other.C_b &&
        C_c == other.C_c &&
        C_n == other.C_n &&
        C_p == other.C_p &&
        L_a == other.L_a &&
        L_b == other.L_b &&
        L_c == other.L_c &&
        L_g_a == other.L_g_a &&
        L_g_b == other.L_g_b &&
        L_g_c == other.L_g_c &&
        R_D_a_n == other.R_D_a_n &&
        R_D_a_p == other.R_D_a_p &&
        R_D_b_n == other.R_D_b_n &&
        R_D_b_p == other.R_D_b_p &&
        R_D_c_n == other.R_D_c_n &&
        R_D_c_p == other.R_D_c_p &&
        R_a == other.R_a &&
        R_b == other.R_b &&
        R_c == other.R_c &&
        R_dc == other.R_dc &&
        R_g_a == other.R_g_a &&
        R_g_b == other.R_g_b &&
        R_g_c == other.R_g_c &&
        R_n_p == other.R_n_p &&
        R_n_s == other.R_n_s &&
        R_p_p == other.R_p_p &&
        R_p_s == other.R_p_s;
}

uint64_t Model_converter::Components::hash() const {
    uint64_t seed = 0;
    rlc2ss::hash_combine(seed, C_a);
    rlc2ss::hash_combine(seed, C_b);
    rlc2ss::hash_combine(seed, C_c);
    rlc2ss::hash_combine(seed, C_n);
    rlc2ss::hash_combine(seed, C_p);
    rlc2ss::hash_combine(seed, L_a);
    rlc2ss::hash_combine(seed, L_b);
    rlc2ss::hash_combine(seed, L_c);
    rlc2ss::hash_combine(seed, L_g_a);
    rlc2ss::hash_combine(seed, L_g_b);
    rlc2ss::hash_combine(seed, L_g_c);
    rlc2ss::hash_combine(seed, R_D_a_n);
    rlc2ss::hash_combine(seed, R_D_a_p);
    rlc2ss::hash_combine(seed, R_D_b_n);
    rlc2ss::hash_combine(seed, R_D_b_p);
    rlc2ss::hash_combine(seed, R_D_c_n);
    rlc2ss::hash_combine(seed, R_D_c_p);
    rlc2ss::hash_combine(seed, R_a);
    rlc2ss::hash_combine(seed, R_b);
    rlc2ss::hash_combine(seed, R_c);
    rlc2ss::hash_combine(seed, R_dc);
    rlc2ss::hash_combine(seed, R_g_a);
    rlc2ss::hash_combine(seed, R_g_b);
    rlc2ss::hash_combine(seed, R_g_c);
    rlc2ss::hash_combine(seed, R_n_p);
    rlc2ss::hash_combine(seed, R_n_s);
    rlc2ss::hash_combine(seed, R_p_p);
    rlc2ss::hash_combine(seed, R_p_s);
    return seed;
}

uint64_t Model_converter::Switches::all() const {
    return 0 |
        (uint64_t{S_D_a_n} << 0) |
        (uint64_t{S_D_a_p} << 1) |
        (uint64_t{S_D_b_n} << 2) |
        (uint64_t{S_D_b_p} << 3) |
        (uint64_t{S_D_c_n} << 4) |
        (uint64_t{S_D_c_p} << 5);
}

double Model_converter::Switches::smallestDelay() {
    return std::min({double(rlc2ss::OnOffDelay::MAX_DELAY),
                    S_D_a_n.pendingTime(),
                    S_D_a_p.pendingTime(),
                    S_D_b_n.pendingTime(),
                    S_D_b_p.pendingTime(),
                    S_D_c_n.pendingTime(),
                    S_D_c_p.pendingTime()});
}

void Model_converter::Switches::step(double dt) {
    S_D_a_n.step(dt);
    S_D_a_p.step(dt);
    S_D_b_n.step(dt);
    S_D_b_p.step(dt);
    S_D_c_n.step(dt);
    S_D_c_p.step(dt);
}
