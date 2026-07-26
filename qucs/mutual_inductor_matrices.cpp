
#include "mutual_inductor_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <mutex>
#include <format>
#include <memory>
#include "mutual_inductor_matrices_json.h"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

inline constexpr int MAX_ZERO_CROSS_EVENTS = 100;

static std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices> calcStateSpace(
    Eigen::MatrixXd const& K1,
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

    // Inductor saturation registers zero-crossing callbacks, so the fast path
    // is used only when neither diodes nor saturation need checking.
    if (m_zero_crossing_callbacks.empty()) {
        stepModel(dt);
        return;
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
    states.I_L2 = outputs.I_L2;
    states.I_L3 = outputs.I_L3;
    states.V_Cf = outputs.V_Cf;
}

void Model_mutual_inductor::updateStateSpaceMatrices() {
    static std::mutex            cache_mutex;
    std::scoped_lock<std::mutex> lock(cache_mutex);

    using StateSpaceMap = std::unordered_map<uint64_t, std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices>>;
    static std::unordered_map<uint64_t, StateSpaceMap> state_space_cache;
    uint64_t switch_combination = switches.all();
    uint64_t component_hash = components.hash();
    if (state_space_cache.contains(switch_combination)) {
        std::unordered_map<uint64_t, std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices>>& cache = state_space_cache.at(switch_combination);
        auto it = cache.find(component_hash);
        if (it != cache.end()) {
            m_ss = *it->second;
            return;
        }
    }

    if (m_circuit_json.empty()) {
        m_circuit_json = nlohmann::json::parse(std::string(mutual_inductor_matrices_json_hexdump, mutual_inductor_matrices_json_hexdump + mutual_inductor_matrices_json_hexdump_len));
    }
    assert(m_circuit_json.contains(std::to_string(switches.all())));

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = m_circuit_json[std::to_string(switches.all())].dump();
    s = rlc2ss::replace(s, "Cf", std::format("({})", components.Cf));
    s = rlc2ss::replace(s, "FSRC1", std::format("({})", components.FSRC1));
    s = rlc2ss::replace(s, "K12", std::format("({})", components.K12));
    s = rlc2ss::replace(s, "K21", std::format("({})", components.K21));
    s = rlc2ss::replace(s, "K31", std::format("({})", components.K31));
    s = rlc2ss::replace(s, "L1", std::format("({})", components.L1));
    s = rlc2ss::replace(s, "L2", std::format("({})", components.L2));
    s = rlc2ss::replace(s, "L3", std::format("({})", components.L3));
    s = rlc2ss::replace(s, "R1", std::format("({})", components.R1));
    s = rlc2ss::replace(s, "R2", std::format("({})", components.R2));
    s = rlc2ss::replace(s, "R3", std::format("({})", components.R3));
    s = rlc2ss::replace(s, "R4", std::format("({})", components.R4));

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
