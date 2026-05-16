
#include "converter_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <mutex>
#include <format>
#include <memory>


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
            .event_callback = [&]() {
                switches.S_D_a_n.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_a_n < 0 && switches.S_D_a_n.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_a_n, outputs.I_R_D_a_n),
            .event_callback = [&]() {
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
            .event_callback = [&]() {
                switches.S_D_a_p.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_a_p < 0 && switches.S_D_a_p.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_a_p, outputs.I_R_D_a_p),
            .event_callback = [&]() {
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
            .event_callback = [&]() {
                switches.S_D_b_n.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_b_n < 0 && switches.S_D_b_n.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_b_n, outputs.I_R_D_b_n),
            .event_callback = [&]() {
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
            .event_callback = [&]() {
                switches.S_D_b_p.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_b_p < 0 && switches.S_D_b_p.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_b_p, outputs.I_R_D_b_p),
            .event_callback = [&]() {
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
            .event_callback = [&]() {
                switches.S_D_c_n.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_c_n < 0 && switches.S_D_c_n.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_c_n, outputs.I_R_D_c_n),
            .event_callback = [&]() {
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
            .event_callback = [&]() {
                switches.S_D_c_p.forceOutput(true);
            }
        });
    }
    if (outputs.I_R_D_c_p < 0 && switches.S_D_c_p.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_c_p, outputs.I_R_D_c_p),
            .event_callback = [&]() {
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

void Model_converter::addInductorSaturation(double* inductor, std::vector<double> currents, std::vector<double> inductances) {
    // Check that the currents are ascending and inductances are descending
    assert(currents.size() == inductances.size());
    for (int i = 1; i < currents.size(); ++i) {
        assert(currents[i] >= currents[i - 1]);
        assert(inductances[i] <= inductances[i - 1]);
    }
    int i_L_output_idx = -1;
    if (inductor == &components.L_a) {
        i_L_output_idx = 2;
    }
    if (inductor == &components.L_b) {
        i_L_output_idx = 3;
    }
    if (inductor == &components.L_c) {
        i_L_output_idx = 4;
    }
    if (i_L_output_idx == -1) {
        assert(("Invalid pointer to inductor", false));
    }

    for (int i = 1; i < currents.size(); ++i) {
        double threshold = currents[i];
        double inductance_prev = inductances[i - 1];
        double inductance = inductances[i];
        // Increase inductance when current goes below level
        m_zero_crossing_callbacks.push_back([=](Outputs const& outputs_prev, Outputs const& outputs_new) -> std::optional<rlc2ss::ZeroCrossingEvent> {
            double i_prev = fabs(outputs_prev.data[i_L_output_idx]);
            double i_new = fabs(outputs_new.data[i_L_output_idx]);
            if (i_prev > threshold && i_new < threshold) {
                return rlc2ss::ZeroCrossingEvent{
                    .time = rlc2ss::calcZeroCrossingTime(i_prev - threshold, i_new - threshold),
                    .event_callback = [&]() {
                        *inductor = inductance_prev;
                    }};
            }
            return std::nullopt;
        });
        // Decrease inductance when current goes above level
        m_zero_crossing_callbacks.push_back([=](Outputs const& outputs_prev, Outputs const& outputs_new) -> std::optional<rlc2ss::ZeroCrossingEvent> {
            double i_prev = fabs(outputs_prev.data[i_L_output_idx]);
            double i_new = fabs(outputs_new.data[i_L_output_idx]);
            if (i_prev < threshold && i_new > threshold) {
                return rlc2ss::ZeroCrossingEvent{
                    .time = rlc2ss::calcZeroCrossingTime(i_prev - threshold, i_new - threshold),
                    .event_callback = [&]() {
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

    if (!switches.S_a_p && !switches.S_a_n && outputs.I_L_a > 0) switches.S_D_a_n.forceOutput(true);
    if (!switches.S_a_p && !switches.S_a_n && outputs.I_L_a < 0) switches.S_D_a_p.forceOutput(true);
    if (!switches.S_b_p && !switches.S_b_n && outputs.I_L_b > 0) switches.S_D_b_n.forceOutput(true);
    if (!switches.S_b_p && !switches.S_b_n && outputs.I_L_b < 0) switches.S_D_b_p.forceOutput(true);
    if (!switches.S_c_p && !switches.S_c_n && outputs.I_L_c > 0) switches.S_D_c_n.forceOutput(true);
    if (!switches.S_c_p && !switches.S_c_n && outputs.I_L_c < 0) switches.S_D_c_p.forceOutput(true);

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
        assert(components.C_n != -1);
        assert(components.C_p != -1);
        assert(components.L_a != -1);
        assert(components.L_b != -1);
        assert(components.L_c != -1);
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
        assert(components.R_n_p != -1);
        assert(components.R_n_s != -1);
        assert(components.R_p_p != -1);
        assert(components.R_p_s != -1);
        _M_components_DO_NOT_TOUCH = components;
        _M_switches_DO_NOT_TOUCH = switches;
        updateStateSpaceMatrices();
        m_solver.updateJacobian(m_ss.A);
        // Solve one step with backward euler to reduce numerical oscillations
        m_Bu = m_ss.B * inputs.data;
        if (m_dt_resolution > 0) {
            double multiple = std::round(dt / m_dt_resolution);
            states.data = m_solver.stepBackwardEuler(*this, states.data, 0.0, multiple * m_dt_resolution);
        } else {
            states.data = m_solver.stepBackwardEuler(*this, states.data, 0.0, dt);
        }
    } else {
        m_Bu = m_ss.B * inputs.data;

        if (m_dt_resolution > 0) {
            if (m_dt_correction_mode == TimestepErrorCorrectionMode::NONE) {
                // Solve with tustin as multiples of resolution and ignore any error
                double multiple = std::round(dt / m_dt_resolution);
                states.data = m_solver.stepTustin(*this, states.data, 0.0, multiple * m_dt_resolution);
            } else if (m_dt_correction_mode == TimestepErrorCorrectionMode::ACCUMULATE) {
                // Solve with tustin as multiples of resolution and accumulate error to correct the timestep length
                // on later steps
                double multiple = (dt + m_dt_error_accumulator) / m_dt_resolution;
                m_dt_error_accumulator += dt - std::round(multiple) * m_dt_resolution;
                states.data = m_solver.stepTustin(*this, states.data, 0.0, std::round(multiple) * m_dt_resolution);
            } else if (m_dt_correction_mode == TimestepErrorCorrectionMode::INTEGRATE_ADAPTIVE) {
                // Solve with tustin as multiples of resolution and the remaining time with runge-kutta so
                // that the matrix inverses required for implicit integration can be cached for common timesteps
                // and weird small remainders are solved with adaptive integration.
                double multiple = dt / m_dt_resolution;
                if (std::abs(std::round(multiple) - multiple) > 1e-6) {
                    double dt1 = std::floor(multiple) * m_dt_resolution;
                    double dt2 = (multiple - std::floor(multiple)) * m_dt_resolution;
                    states.data = m_solver.stepTustin(*this, states.data, 0.0, dt1);
                    states.data = m_solver.stepRungeKuttaFehlberg(*this, states.data, 0.0, dt2);
                } else {
                    states.data = m_solver.stepTustin(*this, states.data, 0.0, multiple * m_dt_resolution);
                }
            }
        } else {
            states.data = m_solver.stepTustin(*this, states.data, 0.0, dt);
        }
    }

    // Update output
    outputs.data = m_ss.C * states.data + m_ss.D * inputs.data;

    // Update states from outputs to have correct values for dependent states
    states.I_L_a = outputs.I_L_a;
    states.I_L_b = outputs.I_L_b;
    states.I_L_c = outputs.I_L_c;
    states.V_C_n = outputs.V_C_n;
    states.V_C_p = outputs.V_C_p;
}

void Model_converter::updateStateSpaceMatrices() {
    static std::mutex            cache_mutex;
    std::scoped_lock<std::mutex> lock(cache_mutex);

    using StateSpaceMap = std::unordered_map<uint64_t, std::unique_ptr<Model_converter::StateSpaceMatrices>>;
    static std::unordered_map<uint64_t, StateSpaceMap> state_space_cache;
    uint64_t switch_combination = switches.all();
    uint64_t component_hash = components.hash();
    if (state_space_cache.contains(switch_combination)) {
        std::unordered_map<uint64_t, std::unique_ptr<Model_converter::StateSpaceMatrices>>& cache = state_space_cache.at(switch_combination);
        auto it = cache.find(component_hash);
        if (it != cache.end()) {
            m_ss = *it->second;
            return;
        }
    }
    std::string netlist = "R_p_p 0 N_dc_p 1E3 \nR_n_p N_dc_n 0 1E3 \nV_dc _net0 N_dc_n DC 1 \nD_a_p N_c_a N_dc_p \nD_b_p N_c_b N_dc_p \nS_a_p N_c_a N_dc_p _net1 _net2 \nS_c_p N_c_c N_dc_p _net3 _net4 \nD_c_p N_c_c N_dc_p \nS_b_p N_c_b N_dc_p _net5 _net6 \nS_a_n N_dc_n N_c_a _net7 _net8 \nD_a_n N_dc_n N_c_a \nS_b_n N_dc_n N_c_b _net9 _net10 \nD_b_n N_dc_n N_c_b \nS_c_n N_dc_n N_c_c _net11 _net12 \nD_c_n N_dc_n N_c_c \nV_a _net13 _net14 DC 0 SIN(0 1 1K 0 0 0) AC 1 ACPHASE 0 \nV_c _net15 _net14 DC 0 SIN(0 1 1K 0 0 0) AC 1 ACPHASE 0 \nV_b _net16 _net14 DC 0 SIN(0 1 1K 0 0 0) AC 1 ACPHASE 0 \nL_b _net17 _net16 1M \nL_a _net18 _net13 1M \nL_c _net19 _net15 1M \nC_p 0 _net20 10E-3;I; \nC_n _net21 0 10E-3;I; \nR_n_s N_dc_n _net21 10E-3 \nR_p_s _net20 N_dc_p 10E-3 \nR_a N_c_a _net18 10E-3 \nR_b N_c_b _net17 10E-3 \nR_c N_c_c _net19 10E-3 \nR_dc _net0 N_dc_p 1;I; ";

    // Cache symbolic intermediate matrices per switch combination
    static std::unordered_map<uint64_t, rlc2ss::SymbolicStateSpace> symbolic_cache;
    if (!symbolic_cache.contains(switch_combination)) {
        symbolic_cache[switch_combination] = rlc2ss::formStateSpaceMatrices(netlist, switch_combination);
    }
    rlc2ss::SymbolicStateSpace const& symbolic_ss = symbolic_cache[switch_combination];

    // Substitute component values into cached symbolic matrices via the typed
    // evaluator (memoised DAG walk over the AST nodes, no string parsing).
    std::unordered_map<std::string, double> values{
        {"C_n", components.C_n},
        {"C_p", components.C_p},
        {"L_a", components.L_a},
        {"L_b", components.L_b},
        {"L_c", components.L_c},
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
    m_ss = *state_space_cache[switch_combination][component_hash];
}

bool Model_converter::Components::operator==(Components const& other) const {
    return
        C_n == other.C_n &&
        C_p == other.C_p &&
        L_a == other.L_a &&
        L_b == other.L_b &&
        L_c == other.L_c &&
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
        R_n_p == other.R_n_p &&
        R_n_s == other.R_n_s &&
        R_p_p == other.R_p_p &&
        R_p_s == other.R_p_s;
}

uint64_t Model_converter::Components::hash() const {
    uint64_t seed = 0;
    rlc2ss::hash_combine(seed, C_n);
    rlc2ss::hash_combine(seed, C_p);
    rlc2ss::hash_combine(seed, L_a);
    rlc2ss::hash_combine(seed, L_b);
    rlc2ss::hash_combine(seed, L_c);
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
    rlc2ss::hash_combine(seed, R_n_p);
    rlc2ss::hash_combine(seed, R_n_s);
    rlc2ss::hash_combine(seed, R_p_p);
    rlc2ss::hash_combine(seed, R_p_s);
    return seed;
}

uint64_t Model_converter::Switches::all() const {
    return 0 |
        (S_D_a_n << 0) |
        (S_D_a_p << 1) |
        (S_D_b_n << 2) |
        (S_D_b_p << 3) |
        (S_D_c_n << 4) |
        (S_D_c_p << 5) |
        (S_a_n << 6) |
        (S_a_p << 7) |
        (S_b_n << 8) |
        (S_b_p << 9) |
        (S_c_n << 10) |
        (S_c_p << 11);
}

double Model_converter::Switches::smallestDelay() {
    return std::min({double(rlc2ss::OnOffDelay::MAX_DELAY),
                    S_D_a_n.pendingTime(),
                    S_D_a_p.pendingTime(),
                    S_D_b_n.pendingTime(),
                    S_D_b_p.pendingTime(),
                    S_D_c_n.pendingTime(),
                    S_D_c_p.pendingTime(),
                    S_a_n.pendingTime(),
                    S_a_p.pendingTime(),
                    S_b_n.pendingTime(),
                    S_b_p.pendingTime(),
                    S_c_n.pendingTime(),
                    S_c_p.pendingTime()});
}

void Model_converter::Switches::step(double dt) {
    S_D_a_n.step(dt);
    S_D_a_p.step(dt);
    S_D_b_n.step(dt);
    S_D_b_p.step(dt);
    S_D_c_n.step(dt);
    S_D_c_p.step(dt);
    S_a_n.step(dt);
    S_a_p.step(dt);
    S_b_n.step(dt);
    S_b_p.step(dt);
    S_c_n.step(dt);
    S_c_p.step(dt);
}
