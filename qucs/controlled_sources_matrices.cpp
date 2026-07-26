
#include "controlled_sources_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <mutex>
#include <format>
#include <memory>
#include "controlled_sources_matrices_json.h"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

inline constexpr int MAX_ZERO_CROSS_EVENTS = 100;

static std::unique_ptr<Model_controlled_sources::StateSpaceMatrices> calcStateSpace(
    Eigen::MatrixXd const& K1,
    Eigen::MatrixXd const& A1,
    Eigen::MatrixXd const& B1,
    Eigen::MatrixXd const& K2,
    Eigen::MatrixXd const& C1,
    Eigen::MatrixXd const& D1) {
    auto ss = std::make_unique<Model_controlled_sources::StateSpaceMatrices>();
    Eigen::MatrixXd A = K1.partialPivLu().solve(A1);
    Eigen::MatrixXd B = K1.partialPivLu().solve(B1);
    ss->A = A;
    ss->B = B;
    ss->C = (C1 + K2 * A);
    ss->D = (D1 + K2 * B);
    return ss;
}

std::optional<rlc2ss::ZeroCrossingEvent> Model_controlled_sources::checkZeroCrossingEvents(Model_controlled_sources::Outputs const& prev_outputs) {
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

Model_controlled_sources::Model_controlled_sources(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
}

void Model_controlled_sources::addInductorSaturation(double* inductor, std::vector<double> currents, std::vector<double> inductances) {
    // Check that the currents are ascending and inductances are descending
    assert(currents.size() == inductances.size());
    for (int i = 1; i < currents.size(); ++i) {
        assert(currents[i] >= currents[i - 1]);
        assert(inductances[i] <= inductances[i - 1]);
    }
    int i_L_output_idx = -1;
    if (inductor == &components.L1) {
        i_L_output_idx = 1;
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

void Model_controlled_sources::step(double dt, Inputs const& inputs_) {
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

void Model_controlled_sources::stepWithZeroCrossingDetection(double dt) {
    // No need to do anything
    if (dt < rlc2ss::MINIMUM_TIMESTEP) {
        return;
    }

    // Inductor saturation registers zero-crossing callbacks, so the fast path
    // is used only when neither diodes nor saturation need checking.
    if (m_zero_crossing_callbacks.empty()) {
        stepModel(dt);
        return;
    }

    // Copy previous state and outputs if step needs to be redone
    Model_controlled_sources::States prev_state;
    Model_controlled_sources::Outputs prev_outputs;
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

void Model_controlled_sources::stepModel(double dt) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all() != _M_switches_DO_NOT_TOUCH.all() || !m_solver.initialized()) {
        assert(components.C_1 != -1);
        assert(components.C_2 != -1);
        assert(components.ESRC3 != -1);
        assert(components.FSRC5 != -1);
        assert(components.GSRC1 != -1);
        assert(components.HSRC4 != -1);
        assert(components.L1 != -1);
        assert(components.R1 != -1);
        assert(components.R2 != -1);
        assert(components.R3 != -1);
        assert(components.R4 != -1);
        assert(components.R5 != -1);
        assert(components.R6 != -1);
        assert(components.R7 != -1);
        _M_components_DO_NOT_TOUCH = components;
        _M_switches_DO_NOT_TOUCH = switches;
        updateStateSpaceMatrices();
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
    states.V_C_1 = outputs.V_C_1;
    states.V_C_2 = outputs.V_C_2;
}

void Model_controlled_sources::updateStateSpaceMatrices() {
    static std::mutex            cache_mutex;
    std::scoped_lock<std::mutex> lock(cache_mutex);

    using StateSpaceMap = std::unordered_map<uint64_t, std::unique_ptr<Model_controlled_sources::StateSpaceMatrices>>;
    static std::unordered_map<uint64_t, StateSpaceMap> state_space_cache;
    uint64_t switch_combination = switches.all();
    uint64_t component_hash = components.hash();
    if (state_space_cache.contains(switch_combination)) {
        std::unordered_map<uint64_t, std::unique_ptr<Model_controlled_sources::StateSpaceMatrices>>& cache = state_space_cache.at(switch_combination);
        auto it = cache.find(component_hash);
        if (it != cache.end()) {
            m_ss = *it->second;
            return;
        }
    }

    if (m_circuit_json.empty()) {
        m_circuit_json = nlohmann::json::parse(std::string(controlled_sources_matrices_json_hexdump, controlled_sources_matrices_json_hexdump + controlled_sources_matrices_json_hexdump_len));
    }
    assert(m_circuit_json.contains(std::to_string(switches.all())));

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = m_circuit_json[std::to_string(switches.all())].dump();
    s = rlc2ss::replace(s, "C_1", std::format("({})", components.C_1));
    s = rlc2ss::replace(s, "C_2", std::format("({})", components.C_2));
    s = rlc2ss::replace(s, "ESRC3", std::format("({})", components.ESRC3));
    s = rlc2ss::replace(s, "FSRC5", std::format("({})", components.FSRC5));
    s = rlc2ss::replace(s, "GSRC1", std::format("({})", components.GSRC1));
    s = rlc2ss::replace(s, "HSRC4", std::format("({})", components.HSRC4));
    s = rlc2ss::replace(s, "L1", std::format("({})", components.L1));
    s = rlc2ss::replace(s, "R1", std::format("({})", components.R1));
    s = rlc2ss::replace(s, "R2", std::format("({})", components.R2));
    s = rlc2ss::replace(s, "R3", std::format("({})", components.R3));
    s = rlc2ss::replace(s, "R4", std::format("({})", components.R4));
    s = rlc2ss::replace(s, "R5", std::format("({})", components.R5));
    s = rlc2ss::replace(s, "R6", std::format("({})", components.R6));
    s = rlc2ss::replace(s, "R7", std::format("({})", components.R7));

    // Parse json for the intermediate matrices
    nlohmann::json j = nlohmann::json::parse(s);
    rlc2ss::StateSpaceMatrices ss = {
        .K1 = j["K1"],
        .K2 = j["K2"],
        .A1 = j["A1"],
        .B1 = j["B1"],
        .C1 = j["C1"],
        .D1 = j["D1"],
    };
    // Create eigen matrices
    Eigen::Matrix<double, NUM_STATES, NUM_STATES, Eigen::RowMajor> K1(rlc2ss::getCommaDelimitedValues(ss.K1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES, Eigen::RowMajor> K2(rlc2ss::getCommaDelimitedValues(ss.K2).data());
    Eigen::Matrix<double, NUM_STATES, NUM_STATES, Eigen::RowMajor> A1(rlc2ss::getCommaDelimitedValues(ss.A1).data());
    Eigen::Matrix<double, NUM_STATES, NUM_INPUTS, Eigen::RowMajor> B1(rlc2ss::getCommaDelimitedValues(ss.B1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES, Eigen::RowMajor> C1(rlc2ss::getCommaDelimitedValues(ss.C1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_INPUTS, Eigen::RowMajor> D1(rlc2ss::getCommaDelimitedValues(ss.D1).data());

    state_space_cache[switch_combination][component_hash] = calcStateSpace(K1, A1, B1, K2, C1, D1);
    m_ss = *state_space_cache[switch_combination][component_hash];
}

bool Model_controlled_sources::Components::operator==(Components const& other) const {
    return
        C_1 == other.C_1 &&
        C_2 == other.C_2 &&
        ESRC3 == other.ESRC3 &&
        FSRC5 == other.FSRC5 &&
        GSRC1 == other.GSRC1 &&
        HSRC4 == other.HSRC4 &&
        L1 == other.L1 &&
        R1 == other.R1 &&
        R2 == other.R2 &&
        R3 == other.R3 &&
        R4 == other.R4 &&
        R5 == other.R5 &&
        R6 == other.R6 &&
        R7 == other.R7;
}

uint64_t Model_controlled_sources::Components::hash() const {
    uint64_t seed = 0;
    rlc2ss::hash_combine(seed, C_1);
    rlc2ss::hash_combine(seed, C_2);
    rlc2ss::hash_combine(seed, ESRC3);
    rlc2ss::hash_combine(seed, FSRC5);
    rlc2ss::hash_combine(seed, GSRC1);
    rlc2ss::hash_combine(seed, HSRC4);
    rlc2ss::hash_combine(seed, L1);
    rlc2ss::hash_combine(seed, R1);
    rlc2ss::hash_combine(seed, R2);
    rlc2ss::hash_combine(seed, R3);
    rlc2ss::hash_combine(seed, R4);
    rlc2ss::hash_combine(seed, R5);
    rlc2ss::hash_combine(seed, R6);
    rlc2ss::hash_combine(seed, R7);
    return seed;
}

uint64_t Model_controlled_sources::Switches::all() const {
    return 0;
}

double Model_controlled_sources::Switches::smallestDelay() {
    return std::min({double(rlc2ss::OnOffDelay::MAX_DELAY),
                    });
}

void Model_controlled_sources::Switches::step(double dt) {
    
}
