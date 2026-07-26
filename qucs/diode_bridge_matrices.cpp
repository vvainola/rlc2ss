
#include "diode_bridge_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <mutex>
#include <format>
#include <memory>
#include "diode_bridge_matrices_json.h"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

inline constexpr int MAX_ZERO_CROSS_EVENTS = 100;

static std::unique_ptr<Model_diode_bridge::StateSpaceMatrices> calcStateSpace(
    Eigen::MatrixXd const& K1,
    Eigen::MatrixXd const& A1,
    Eigen::MatrixXd const& B1,
    Eigen::MatrixXd const& K2,
    Eigen::MatrixXd const& C1,
    Eigen::MatrixXd const& D1) {
    auto ss = std::make_unique<Model_diode_bridge::StateSpaceMatrices>();
    Eigen::MatrixXd A = K1.partialPivLu().solve(A1);
    Eigen::MatrixXd B = K1.partialPivLu().solve(B1);
    ss->A = A;
    ss->B = B;
    ss->C = (C1 + K2 * A);
    ss->D = (D1 + K2 * B);
    return ss;
}

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
    if (outputs.I_R_D_n_a < 0 && switches.S_D_n_a.outputForced()) {
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
    if (outputs.I_R_D_n_b < 0 && switches.S_D_n_b.outputForced()) {
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
    if (outputs.I_R_D_n_c < 0 && switches.S_D_n_c.outputForced()) {
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
    if (outputs.I_R_D_p_a < 0 && switches.S_D_p_a.outputForced()) {
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
    if (outputs.I_R_D_p_b < 0 && switches.S_D_p_b.outputForced()) {
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
    if (outputs.I_R_D_p_c < 0 && switches.S_D_p_c.outputForced()) {
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

void Model_diode_bridge::stepWithZeroCrossingDetection(double dt) {
    // No need to do anything
    if (dt < rlc2ss::MINIMUM_TIMESTEP) {
        return;
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
    dt = std::max(dt, m_dt_resolution);
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
    states.I_L_a = outputs.I_L_a;
    states.I_L_b = outputs.I_L_b;
    states.I_L_c = outputs.I_L_c;
    states.V_C_dc = outputs.V_C_dc;
}

void Model_diode_bridge::updateStateSpaceMatrices() {
    static std::mutex            cache_mutex;
    std::scoped_lock<std::mutex> lock(cache_mutex);

    using StateSpaceMap = std::unordered_map<uint64_t, std::unique_ptr<Model_diode_bridge::StateSpaceMatrices>>;
    static std::unordered_map<uint64_t, StateSpaceMap> state_space_cache;
    uint64_t switch_combination = switches.all();
    uint64_t component_hash = components.hash();
    if (state_space_cache.contains(switch_combination)) {
        std::unordered_map<uint64_t, std::unique_ptr<Model_diode_bridge::StateSpaceMatrices>>& cache = state_space_cache.at(switch_combination);
        auto it = cache.find(component_hash);
        if (it != cache.end()) {
            m_ss = *it->second;
            return;
        }
    }

    if (m_circuit_json.empty()) {
        m_circuit_json = nlohmann::json::parse(std::string(diode_bridge_matrices_json_hexdump, diode_bridge_matrices_json_hexdump + diode_bridge_matrices_json_hexdump_len));
    }
    assert(m_circuit_json.contains(std::to_string(switches.all())));

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = m_circuit_json[std::to_string(switches.all())].dump();
    s = rlc2ss::replace(s, "C_dc", std::format("({})", components.C_dc));
    s = rlc2ss::replace(s, "L_a", std::format("({})", components.L_a));
    s = rlc2ss::replace(s, "L_b", std::format("({})", components.L_b));
    s = rlc2ss::replace(s, "L_c", std::format("({})", components.L_c));
    s = rlc2ss::replace(s, "R_D_n_a", std::format("({})", components.R_D_n_a));
    s = rlc2ss::replace(s, "R_D_n_b", std::format("({})", components.R_D_n_b));
    s = rlc2ss::replace(s, "R_D_n_c", std::format("({})", components.R_D_n_c));
    s = rlc2ss::replace(s, "R_D_p_a", std::format("({})", components.R_D_p_a));
    s = rlc2ss::replace(s, "R_D_p_b", std::format("({})", components.R_D_p_b));
    s = rlc2ss::replace(s, "R_D_p_c", std::format("({})", components.R_D_p_c));
    s = rlc2ss::replace(s, "R_a", std::format("({})", components.R_a));
    s = rlc2ss::replace(s, "R_b", std::format("({})", components.R_b));
    s = rlc2ss::replace(s, "R_c", std::format("({})", components.R_c));
    s = rlc2ss::replace(s, "R_dc", std::format("({})", components.R_dc));
    s = rlc2ss::replace(s, "R_load", std::format("({})", components.R_load));

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
        (S_D_n_a << 0) |
        (S_D_n_b << 1) |
        (S_D_n_c << 2) |
        (S_D_p_a << 3) |
        (S_D_p_b << 4) |
        (S_D_p_c << 5);
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
