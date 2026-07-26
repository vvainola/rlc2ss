
#include "mutual_inductor_matrices.hpp"
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

std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices> calcStateSpace(Eigen::MatrixXd const& K1,
                                                                          Eigen::MatrixXd const& A1,
                                                                          Eigen::MatrixXd const& B1,
                                                                          Eigen::MatrixXd const& K2,
                                                                          Eigen::MatrixXd const& C1,
                                                                          Eigen::MatrixXd const& D1) {
    auto ss = std::make_unique<Model_mutual_inductor::StateSpaceMatrices>();
    auto lu = K1.partialPivLu();
    Eigen::MatrixXd A = lu.solve(A1);
    Eigen::MatrixXd B = lu.solve(B1);
    ss->A = A;
    ss->B = B;
    ss->C = (C1 + K2 * A);
    ss->D = (D1 + K2 * B);
    return ss;
}

Model_mutual_inductor::StateSpaceMatrices const& calcStateSpaceMatrices(Model_mutual_inductor::Components const& components,
                                                                        uint64_t switch_combination) {
    static std::mutex            cache_mutex;
    std::scoped_lock<std::mutex> lock(cache_mutex);

    using StateSpaceMap = std::unordered_map<uint64_t, std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices>>;
    static std::unordered_map<uint64_t, StateSpaceMap> state_space_cache;
    uint64_t component_hash = components.hash();
    if (state_space_cache.contains(switch_combination)) {
        std::unordered_map<uint64_t, std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices>>& cache = state_space_cache.at(switch_combination);
        auto it = cache.find(component_hash);
        if (it != cache.end()) {
            return *it->second;
        }
    }
    std::string netlist = "V1 _net0 0 DC 1 \nV2 _net1 0 DC 1 \nL1 N1 _net2 1; \nL2 N2 _net2 1; \nL3 N3 _net2 1; \nK12 L1 L2 0.9; \nK21 L2 L3 0.9; \nK31 L3 L1 0.9; \nR2 _net1 N2 10; \nR1 _net0 N1 10; \nR3 _net3 N3 10E-3;I; \nV3 _net3 0 DC 1I; \nR4 _net4 _net5 10;I; \nFSRC1 _net4 0 VSRC1 10; \nVSRC1 _net2 0 DC 0 \nCf 0 _net5 100E-6; ";

    // Cache symbolic intermediate matrices per switch combination
    static std::unordered_map<uint64_t, rlc2ss::SymbolicStateSpace> symbolic_cache;
    if (!symbolic_cache.contains(switch_combination)) {
        symbolic_cache[switch_combination] = rlc2ss::formStateSpaceMatrices(netlist, switch_combination);
    }
    rlc2ss::SymbolicStateSpace const& symbolic_ss = symbolic_cache[switch_combination];

    // Substitute component values into cached symbolic matrices
    std::unordered_map<std::string, double> values{
        {"Cf", components.Cf},
        {"FSRC1", components.FSRC1},
        {"K12", components.K12},
        {"K21", components.K21},
        {"K31", components.K31},
        {"L1", components.L1},
        {"L2", components.L2},
        {"L3", components.L3},
        {"R1", components.R1},
        {"R2", components.R2},
        {"R3", components.R3},
        {"R4", components.R4},
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


Model_mutual_inductor::Outputs calcInstantaneousOutputs(Model_mutual_inductor::Components const& components,
                                                        Model_mutual_inductor::States const& states,
                                                        Model_mutual_inductor::Inputs const& inputs,
                                                        uint64_t switch_combination) {
    Model_mutual_inductor::Outputs instantaneous_outputs;
    auto const& ss = calcStateSpaceMatrices(components, switch_combination);
    instantaneous_outputs.data = ss.C * states.data + ss.D * inputs.data;
    return instantaneous_outputs;
}

uint64_t externalClosedSwitchMask(Model_mutual_inductor::Switches const& switches) {
    return 0;
}

Model_mutual_inductor::Switches releaseReverseCurrentDiodes(Model_mutual_inductor::Components const&,
                                                            Model_mutual_inductor::States const&,
                                                            Model_mutual_inductor::Inputs const&,
                                                            Model_mutual_inductor::Switches const& switches) {
    return switches;
}

Model_mutual_inductor::Switches resolveDiodeContinuity(Model_mutual_inductor::Components const&,
                                                       Model_mutual_inductor::States const&,
                                                       Model_mutual_inductor::Inputs const&,
                                                       Model_mutual_inductor::Switches const& switches,
                                                       uint64_t) {
    return switches;
}


} // namespace

std::optional<rlc2ss::ZeroCrossingEvent> Model_mutual_inductor::checkZeroCrossingEvents(Model_mutual_inductor::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

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

Model_mutual_inductor::Model_mutual_inductor(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
}



void Model_mutual_inductor::addInductorSaturation(double* inductor, std::vector<double> currents, std::vector<double> inductances) {
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

void Model_mutual_inductor::step(double dt, Inputs const& inputs_) {
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

void Model_mutual_inductor::stepWithZeroCrossingDetection(double dt) {
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
    Model_mutual_inductor::States prev_state;
    Model_mutual_inductor::Outputs prev_outputs;
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

void Model_mutual_inductor::stepModel(double dt) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all() != _M_switches_DO_NOT_TOUCH.all() || !m_solver.initialized()) {
        assert(components.Cf != -1);
        assert(components.FSRC1 != -1);
        assert(components.K12 != -1);
        assert(components.K21 != -1);
        assert(components.K31 != -1);
        assert(components.L1 != -1);
        assert(components.L2 != -1);
        assert(components.L3 != -1);
        assert(components.R1 != -1);
        assert(components.R2 != -1);
        assert(components.R3 != -1);
        assert(components.R4 != -1);
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
    states.V_Cf = outputs.V_Cf;
}

bool Model_mutual_inductor::Components::operator==(Components const& other) const {
    return
        Cf == other.Cf &&
        FSRC1 == other.FSRC1 &&
        K12 == other.K12 &&
        K21 == other.K21 &&
        K31 == other.K31 &&
        L1 == other.L1 &&
        L2 == other.L2 &&
        L3 == other.L3 &&
        R1 == other.R1 &&
        R2 == other.R2 &&
        R3 == other.R3 &&
        R4 == other.R4;
}

uint64_t Model_mutual_inductor::Components::hash() const {
    uint64_t seed = 0;
    rlc2ss::hash_combine(seed, Cf);
    rlc2ss::hash_combine(seed, FSRC1);
    rlc2ss::hash_combine(seed, K12);
    rlc2ss::hash_combine(seed, K21);
    rlc2ss::hash_combine(seed, K31);
    rlc2ss::hash_combine(seed, L1);
    rlc2ss::hash_combine(seed, L2);
    rlc2ss::hash_combine(seed, L3);
    rlc2ss::hash_combine(seed, R1);
    rlc2ss::hash_combine(seed, R2);
    rlc2ss::hash_combine(seed, R3);
    rlc2ss::hash_combine(seed, R4);
    return seed;
}

uint64_t Model_mutual_inductor::Switches::all() const {
    return 0;
}

double Model_mutual_inductor::Switches::smallestDelay() {
    return std::min({double(rlc2ss::OnOffDelay::MAX_DELAY),
                    });
}

void Model_mutual_inductor::Switches::step(double dt) {
    
}
