
#include "diode_bridge_3l_matrices.hpp"
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

static std::unique_ptr<Model_diode_bridge_3l::StateSpaceMatrices> calcStateSpace(
    Eigen::MatrixXd const& K1,
    Eigen::MatrixXd const& A1,
    Eigen::MatrixXd const& B1,
    Eigen::MatrixXd const& K2,
    Eigen::MatrixXd const& C1,
    Eigen::MatrixXd const& D1) {
    auto ss = std::make_unique<Model_diode_bridge_3l::StateSpaceMatrices>();
    Eigen::MatrixXd A = K1.partialPivLu().solve(A1);
    Eigen::MatrixXd B = K1.partialPivLu().solve(B1);
    ss->A = A;
    ss->B = B;
    ss->C = (C1 + K2 * A);
    ss->D = (D1 + K2 * B);
    return ss;
}

std::optional<rlc2ss::ZeroCrossingEvent> Model_diode_bridge_3l::checkZeroCrossingEvents(Model_diode_bridge_3l::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    // Diode D_n_a
    double V_D_n_a = outputs.N_dc_n - outputs.N_conv_a;
    if (V_D_n_a > inputs.V_D_n_a && !switches.S_D_n_a) {
        double V_D_n_a_prev = prev_outputs.N_dc_n - prev_outputs.N_conv_a;
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
    double V_D_n_b = outputs.N_dc_n - outputs.N_conv_b;
    if (V_D_n_b > inputs.V_D_n_b && !switches.S_D_n_b) {
        double V_D_n_b_prev = prev_outputs.N_dc_n - prev_outputs.N_conv_b;
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
    double V_D_n_c = outputs.N_dc_n - outputs.N_conv_c;
    if (V_D_n_c > inputs.V_D_n_c && !switches.S_D_n_c) {
        double V_D_n_c_prev = prev_outputs.N_dc_n - prev_outputs.N_conv_c;
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
    double V_D_p_a = outputs.N_conv_a - outputs.N_dc_p;
    if (V_D_p_a > inputs.V_D_p_a && !switches.S_D_p_a) {
        double V_D_p_a_prev = prev_outputs.N_conv_a - prev_outputs.N_dc_p;
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
    double V_D_p_b = outputs.N_conv_b - outputs.N_dc_p;
    if (V_D_p_b > inputs.V_D_p_b && !switches.S_D_p_b) {
        double V_D_p_b_prev = prev_outputs.N_conv_b - prev_outputs.N_dc_p;
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
    double V_D_p_c = outputs.N_conv_c - outputs.N_dc_p;
    if (V_D_p_c > inputs.V_D_p_c && !switches.S_D_p_c) {
        double V_D_p_c_prev = prev_outputs.N_conv_c - prev_outputs.N_dc_p;
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

Model_diode_bridge_3l::Model_diode_bridge_3l(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
}

void Model_diode_bridge_3l::addInductorSaturation(double* inductor, std::vector<double> currents, std::vector<double> inductances) {
    // Check that the currents are ascending and inductances are descending
    assert(currents.size() == inductances.size());
    for (int i = 1; i < currents.size(); ++i) {
        assert(currents[i] >= currents[i - 1]);
        assert(inductances[i] <= inductances[i - 1]);
    }
    int i_L_output_idx = -1;
    if (inductor == &components.L_conv_a) {
        i_L_output_idx = 0;
    }
    if (inductor == &components.L_conv_b) {
        i_L_output_idx = 1;
    }
    if (inductor == &components.L_conv_c) {
        i_L_output_idx = 2;
    }
    if (inductor == &components.L_dc_n) {
        i_L_output_idx = 3;
    }
    if (inductor == &components.L_dc_p) {
        i_L_output_idx = 4;
    }
    if (inductor == &components.L_dc_src) {
        i_L_output_idx = 5;
    }
    if (inductor == &components.L_grid_a) {
        i_L_output_idx = 6;
    }
    if (inductor == &components.L_grid_b) {
        i_L_output_idx = 7;
    }
    if (inductor == &components.L_grid_c) {
        i_L_output_idx = 8;
    }
    if (inductor == &components.L_src_a) {
        i_L_output_idx = 9;
    }
    if (inductor == &components.L_src_b) {
        i_L_output_idx = 10;
    }
    if (inductor == &components.L_src_c) {
        i_L_output_idx = 11;
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

void Model_diode_bridge_3l::step(double dt, Inputs const& inputs_) {
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

void Model_diode_bridge_3l::stepWithZeroCrossingDetection(double dt) {
    // No need to do anything
    if (dt < rlc2ss::MINIMUM_TIMESTEP) {
        return;
    }

    // Copy previous state and outputs if step needs to be redone
    Model_diode_bridge_3l::States prev_state;
    Model_diode_bridge_3l::Outputs prev_outputs;
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

void Model_diode_bridge_3l::stepModel(double dt) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all() != _M_switches_DO_NOT_TOUCH.all() || !m_solver.initialized()) {
        assert(components.C_dc_n1 != -1);
        assert(components.C_dc_n2 != -1);
        assert(components.C_dc_p1 != -1);
        assert(components.C_dc_p2 != -1);
        assert(components.C_f_a != -1);
        assert(components.C_f_b != -1);
        assert(components.C_f_c != -1);
        assert(components.L_conv_a != -1);
        assert(components.L_conv_b != -1);
        assert(components.L_conv_c != -1);
        assert(components.L_dc_n != -1);
        assert(components.L_dc_p != -1);
        assert(components.L_dc_src != -1);
        assert(components.L_grid_a != -1);
        assert(components.L_grid_b != -1);
        assert(components.L_grid_c != -1);
        assert(components.L_src_a != -1);
        assert(components.L_src_b != -1);
        assert(components.L_src_c != -1);
        assert(components.R_D_n_a != -1);
        assert(components.R_D_n_b != -1);
        assert(components.R_D_n_c != -1);
        assert(components.R_D_p_a != -1);
        assert(components.R_D_p_b != -1);
        assert(components.R_D_p_c != -1);
        assert(components.R_conv_a != -1);
        assert(components.R_conv_b != -1);
        assert(components.R_conv_c != -1);
        assert(components.R_dc_pn1 != -1);
        assert(components.R_dc_pn2 != -1);
        assert(components.R_dc_pp1 != -1);
        assert(components.R_dc_pp2 != -1);
        assert(components.R_dc_sn1 != -1);
        assert(components.R_dc_sn2 != -1);
        assert(components.R_dc_sp1 != -1);
        assert(components.R_dc_sp2 != -1);
        assert(components.R_dc_src_p != -1);
        assert(components.R_dc_src_s != -1);
        assert(components.R_f_a != -1);
        assert(components.R_f_b != -1);
        assert(components.R_f_c != -1);
        assert(components.R_grid_a != -1);
        assert(components.R_grid_b != -1);
        assert(components.R_grid_c != -1);
        assert(components.R_src_a != -1);
        assert(components.R_src_b != -1);
        assert(components.R_src_c != -1);
        _M_components_DO_NOT_TOUCH = components;
        _M_switches_DO_NOT_TOUCH = switches;
        updateStateSpaceMatrices();
        m_solver.updateJacobian(m_ss.A);
        // Solve one step with backward euler to reduce numerical oscillations
        m_Bu = m_ss.B * inputs.data;
        if (m_dt_resolution > 0) {
            double multiple = std::round(dt / m_dt_resolution);
            states.data = m_solver.stepLinearBackwardEuler(states.data, m_Bu, multiple * m_dt_resolution);
        } else {
            states.data = m_solver.stepLinearBackwardEuler(states.data, m_Bu, dt);
        }
    } else {
        m_Bu = m_ss.B * inputs.data;

        if (m_dt_resolution > 0) {
            if (m_dt_correction_mode == TimestepErrorCorrectionMode::NONE) {
                // Solve with tustin as multiples of resolution and ignore any error
                double multiple = std::round(dt / m_dt_resolution);
                states.data = m_solver.stepLinearTustin(states.data, m_Bu, multiple * m_dt_resolution);
            } else if (m_dt_correction_mode == TimestepErrorCorrectionMode::ACCUMULATE) {
                // Solve with tustin as multiples of resolution and accumulate error to correct the timestep length
                // on later steps
                double multiple = (dt + m_dt_error_accumulator) / m_dt_resolution;
                m_dt_error_accumulator += dt - std::round(multiple) * m_dt_resolution;
                states.data = m_solver.stepLinearTustin(states.data, m_Bu, std::round(multiple) * m_dt_resolution);
            } else if (m_dt_correction_mode == TimestepErrorCorrectionMode::INTEGRATE_ADAPTIVE) {
                // Solve with tustin as multiples of resolution and the remaining time with runge-kutta so
                // that the matrix inverses required for implicit integration can be cached for common timesteps
                // and weird small remainders are solved with adaptive integration.
                double multiple = dt / m_dt_resolution;
                if (std::abs(std::round(multiple) - multiple) > 1e-6) {
                    double dt1 = std::floor(multiple) * m_dt_resolution;
                    double dt2 = (multiple - std::floor(multiple)) * m_dt_resolution;
                    states.data = m_solver.stepLinearTustin(states.data, m_Bu, dt1);
                    states.data = m_solver.stepRungeKuttaFehlberg(*this, states.data, 0.0, dt2);
                } else {
                    states.data = m_solver.stepLinearTustin(states.data, m_Bu, multiple * m_dt_resolution);
                }
            }
        } else {
            states.data = m_solver.stepLinearTustin(states.data, m_Bu, dt);
        }
    }

    // Update output
    outputs.data = m_ss.C * states.data + m_ss.D * inputs.data;

    // Update states from outputs to have correct values for dependent states
    states.I_L_conv_a = outputs.I_L_conv_a;
    states.I_L_conv_b = outputs.I_L_conv_b;
    states.I_L_conv_c = outputs.I_L_conv_c;
    states.I_L_dc_n = outputs.I_L_dc_n;
    states.I_L_dc_p = outputs.I_L_dc_p;
    states.I_L_dc_src = outputs.I_L_dc_src;
    states.I_L_grid_a = outputs.I_L_grid_a;
    states.I_L_grid_b = outputs.I_L_grid_b;
    states.I_L_grid_c = outputs.I_L_grid_c;
    states.I_L_src_a = outputs.I_L_src_a;
    states.I_L_src_b = outputs.I_L_src_b;
    states.I_L_src_c = outputs.I_L_src_c;
    states.V_C_dc_n1 = outputs.V_C_dc_n1;
    states.V_C_dc_n2 = outputs.V_C_dc_n2;
    states.V_C_dc_p1 = outputs.V_C_dc_p1;
    states.V_C_dc_p2 = outputs.V_C_dc_p2;
    states.V_C_f_a = outputs.V_C_f_a;
    states.V_C_f_b = outputs.V_C_f_b;
    states.V_C_f_c = outputs.V_C_f_c;
}

void Model_diode_bridge_3l::updateStateSpaceMatrices() {
    static std::mutex            cache_mutex;
    std::scoped_lock<std::mutex> lock(cache_mutex);

    using StateSpaceMap = std::unordered_map<uint64_t, std::unique_ptr<Model_diode_bridge_3l::StateSpaceMatrices>>;
    static std::unordered_map<uint64_t, StateSpaceMap> state_space_cache;
    uint64_t switch_combination = switches.all();
    uint64_t component_hash = components.hash();
    if (state_space_cache.contains(switch_combination)) {
        std::unordered_map<uint64_t, std::unique_ptr<Model_diode_bridge_3l::StateSpaceMatrices>>& cache = state_space_cache.at(switch_combination);
        auto it = cache.find(component_hash);
        if (it != cache.end()) {
            m_ss = *it->second;
            return;
        }
    }
    std::string netlist = "V_src_a _net0 0 DC 1 \nV_src_b _net1 0 DC 1 \nV_src_c _net2 0 DC 1 \nR_conv_a N_conv_a V2_a 1E-3 \nR_conv_b N_conv_b V2_b 1E-3 \nR_conv_c N_conv_c V2_c 1E-3 \nR_grid_a N_cap_a _net3 1E-3 \nR_grid_b N_cap_b _net4 1E-3 \nR_grid_c N_cap_c _net5 1E-3 \nR_src_a _net6 _net7 1E-3 \nR_src_b _net8 _net9 1E-3 \nR_src_c _net10 _net11 1E-3 \nL_conv_a V2_a N_cap_a 1E-6 \nL_conv_b V2_b N_cap_b 1E-6 \nL_conv_c V2_c N_cap_c 1E-6 \nL_grid_a _net3 _net6 1E-6 \nL_grid_b _net4 _net8 1E-6 \nL_grid_c _net5 _net10 1E-6 \nL_src_a _net7 _net0 1E-6 \nL_src_b _net9 _net1 1E-6 \nL_src_c _net11 _net2 1E-6 \nR_f_a _net12 N_cap_0 1E-3 \nR_f_b _net13 N_cap_0 1E-3 \nR_f_c _net14 N_cap_0 1E-3 \nC_f_a N_cap_a _net12 1E-3 \nC_f_b N_cap_b _net13 1E-3 \nC_f_c N_cap_c _net14 1E-3 \nD_p_c N_conv_c N_dc_p \nD_n_c N_dc_n N_conv_c \nV_dc_src _net15 _net16 DC 1 \nR_dc_src_p _net17 _net18 1E3 \nL_dc_src _net15 _net18 10E-6 \nR_dc_pp2 _net17 _net19 1E3 \nR_dc_pn2 _net19 _net16 1E3 \nR_dc_sp2 _net17 _net20 1E-3 \nR_dc_sn2 _net19 _net21 1E-3 \nR_dc_pp1 N_dc_p N_dc_0 1E3 \nR_dc_pn1 N_dc_0 N_dc_n 1E3 \nR_dc_sp1 N_dc_p _net22 1E-3 \nR_dc_sn1 N_dc_0 _net23 1E-3 \nC_dc_p2 _net20 _net19 10E-3 \nC_dc_p1 _net22 N_dc_0 10E-3 \nC_dc_n1 _net23 N_dc_n 10E-3 \nC_dc_n2 _net21 _net16 10E-3 \nL_dc_n _net16 N_dc_n 1E-6 \nL_dc_p _net17 N_dc_p 1E-6 \nR_dc_src_s _net17 _net18 1 \nS_0_a N_conv_a N_dc_0 _net24 _net25 \nS_0_b N_conv_b N_dc_0 _net26 _net27 \nS_0_c N_conv_c N_dc_0 _net28 _net29 \nD_p_a N_conv_a N_dc_p \nD_n_a N_dc_n N_conv_a \nD_p_b N_conv_b N_dc_p \nD_n_b N_dc_n N_conv_b \nS_p_a N_conv_a N_dc_p _net30 _net31 \nS_p_b N_conv_b N_dc_p _net32 _net33 \nS_p_c N_conv_c N_dc_p _net34 _net35 \nS_n_a N_dc_n N_conv_a _net36 _net37 \nS_n_b N_dc_n N_conv_b _net38 _net39 \nS_n_c N_dc_n N_conv_c _net40 _net41 ";

    // Cache symbolic intermediate matrices per switch combination
    static std::unordered_map<uint64_t, rlc2ss::SymbolicStateSpace> symbolic_cache;
    if (!symbolic_cache.contains(switch_combination)) {
        symbolic_cache[switch_combination] = rlc2ss::formStateSpaceMatrices(netlist, switch_combination);
    }
    rlc2ss::SymbolicStateSpace const& symbolic_ss = symbolic_cache[switch_combination];

    // Substitute component values into cached symbolic matrices
    std::unordered_map<std::string, double> values{
        {"C_dc_n1", components.C_dc_n1},
        {"C_dc_n2", components.C_dc_n2},
        {"C_dc_p1", components.C_dc_p1},
        {"C_dc_p2", components.C_dc_p2},
        {"C_f_a", components.C_f_a},
        {"C_f_b", components.C_f_b},
        {"C_f_c", components.C_f_c},
        {"L_conv_a", components.L_conv_a},
        {"L_conv_b", components.L_conv_b},
        {"L_conv_c", components.L_conv_c},
        {"L_dc_n", components.L_dc_n},
        {"L_dc_p", components.L_dc_p},
        {"L_dc_src", components.L_dc_src},
        {"L_grid_a", components.L_grid_a},
        {"L_grid_b", components.L_grid_b},
        {"L_grid_c", components.L_grid_c},
        {"L_src_a", components.L_src_a},
        {"L_src_b", components.L_src_b},
        {"L_src_c", components.L_src_c},
        {"R_D_n_a", components.R_D_n_a},
        {"R_D_n_b", components.R_D_n_b},
        {"R_D_n_c", components.R_D_n_c},
        {"R_D_p_a", components.R_D_p_a},
        {"R_D_p_b", components.R_D_p_b},
        {"R_D_p_c", components.R_D_p_c},
        {"R_conv_a", components.R_conv_a},
        {"R_conv_b", components.R_conv_b},
        {"R_conv_c", components.R_conv_c},
        {"R_dc_pn1", components.R_dc_pn1},
        {"R_dc_pn2", components.R_dc_pn2},
        {"R_dc_pp1", components.R_dc_pp1},
        {"R_dc_pp2", components.R_dc_pp2},
        {"R_dc_sn1", components.R_dc_sn1},
        {"R_dc_sn2", components.R_dc_sn2},
        {"R_dc_sp1", components.R_dc_sp1},
        {"R_dc_sp2", components.R_dc_sp2},
        {"R_dc_src_p", components.R_dc_src_p},
        {"R_dc_src_s", components.R_dc_src_s},
        {"R_f_a", components.R_f_a},
        {"R_f_b", components.R_f_b},
        {"R_f_c", components.R_f_c},
        {"R_grid_a", components.R_grid_a},
        {"R_grid_b", components.R_grid_b},
        {"R_grid_c", components.R_grid_c},
        {"R_src_a", components.R_src_a},
        {"R_src_b", components.R_src_b},
        {"R_src_c", components.R_src_c},
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

bool Model_diode_bridge_3l::Components::operator==(Components const& other) const {
    return
        C_dc_n1 == other.C_dc_n1 &&
        C_dc_n2 == other.C_dc_n2 &&
        C_dc_p1 == other.C_dc_p1 &&
        C_dc_p2 == other.C_dc_p2 &&
        C_f_a == other.C_f_a &&
        C_f_b == other.C_f_b &&
        C_f_c == other.C_f_c &&
        L_conv_a == other.L_conv_a &&
        L_conv_b == other.L_conv_b &&
        L_conv_c == other.L_conv_c &&
        L_dc_n == other.L_dc_n &&
        L_dc_p == other.L_dc_p &&
        L_dc_src == other.L_dc_src &&
        L_grid_a == other.L_grid_a &&
        L_grid_b == other.L_grid_b &&
        L_grid_c == other.L_grid_c &&
        L_src_a == other.L_src_a &&
        L_src_b == other.L_src_b &&
        L_src_c == other.L_src_c &&
        R_D_n_a == other.R_D_n_a &&
        R_D_n_b == other.R_D_n_b &&
        R_D_n_c == other.R_D_n_c &&
        R_D_p_a == other.R_D_p_a &&
        R_D_p_b == other.R_D_p_b &&
        R_D_p_c == other.R_D_p_c &&
        R_conv_a == other.R_conv_a &&
        R_conv_b == other.R_conv_b &&
        R_conv_c == other.R_conv_c &&
        R_dc_pn1 == other.R_dc_pn1 &&
        R_dc_pn2 == other.R_dc_pn2 &&
        R_dc_pp1 == other.R_dc_pp1 &&
        R_dc_pp2 == other.R_dc_pp2 &&
        R_dc_sn1 == other.R_dc_sn1 &&
        R_dc_sn2 == other.R_dc_sn2 &&
        R_dc_sp1 == other.R_dc_sp1 &&
        R_dc_sp2 == other.R_dc_sp2 &&
        R_dc_src_p == other.R_dc_src_p &&
        R_dc_src_s == other.R_dc_src_s &&
        R_f_a == other.R_f_a &&
        R_f_b == other.R_f_b &&
        R_f_c == other.R_f_c &&
        R_grid_a == other.R_grid_a &&
        R_grid_b == other.R_grid_b &&
        R_grid_c == other.R_grid_c &&
        R_src_a == other.R_src_a &&
        R_src_b == other.R_src_b &&
        R_src_c == other.R_src_c;
}

uint64_t Model_diode_bridge_3l::Components::hash() const {
    uint64_t seed = 0;
    rlc2ss::hash_combine(seed, C_dc_n1);
    rlc2ss::hash_combine(seed, C_dc_n2);
    rlc2ss::hash_combine(seed, C_dc_p1);
    rlc2ss::hash_combine(seed, C_dc_p2);
    rlc2ss::hash_combine(seed, C_f_a);
    rlc2ss::hash_combine(seed, C_f_b);
    rlc2ss::hash_combine(seed, C_f_c);
    rlc2ss::hash_combine(seed, L_conv_a);
    rlc2ss::hash_combine(seed, L_conv_b);
    rlc2ss::hash_combine(seed, L_conv_c);
    rlc2ss::hash_combine(seed, L_dc_n);
    rlc2ss::hash_combine(seed, L_dc_p);
    rlc2ss::hash_combine(seed, L_dc_src);
    rlc2ss::hash_combine(seed, L_grid_a);
    rlc2ss::hash_combine(seed, L_grid_b);
    rlc2ss::hash_combine(seed, L_grid_c);
    rlc2ss::hash_combine(seed, L_src_a);
    rlc2ss::hash_combine(seed, L_src_b);
    rlc2ss::hash_combine(seed, L_src_c);
    rlc2ss::hash_combine(seed, R_D_n_a);
    rlc2ss::hash_combine(seed, R_D_n_b);
    rlc2ss::hash_combine(seed, R_D_n_c);
    rlc2ss::hash_combine(seed, R_D_p_a);
    rlc2ss::hash_combine(seed, R_D_p_b);
    rlc2ss::hash_combine(seed, R_D_p_c);
    rlc2ss::hash_combine(seed, R_conv_a);
    rlc2ss::hash_combine(seed, R_conv_b);
    rlc2ss::hash_combine(seed, R_conv_c);
    rlc2ss::hash_combine(seed, R_dc_pn1);
    rlc2ss::hash_combine(seed, R_dc_pn2);
    rlc2ss::hash_combine(seed, R_dc_pp1);
    rlc2ss::hash_combine(seed, R_dc_pp2);
    rlc2ss::hash_combine(seed, R_dc_sn1);
    rlc2ss::hash_combine(seed, R_dc_sn2);
    rlc2ss::hash_combine(seed, R_dc_sp1);
    rlc2ss::hash_combine(seed, R_dc_sp2);
    rlc2ss::hash_combine(seed, R_dc_src_p);
    rlc2ss::hash_combine(seed, R_dc_src_s);
    rlc2ss::hash_combine(seed, R_f_a);
    rlc2ss::hash_combine(seed, R_f_b);
    rlc2ss::hash_combine(seed, R_f_c);
    rlc2ss::hash_combine(seed, R_grid_a);
    rlc2ss::hash_combine(seed, R_grid_b);
    rlc2ss::hash_combine(seed, R_grid_c);
    rlc2ss::hash_combine(seed, R_src_a);
    rlc2ss::hash_combine(seed, R_src_b);
    rlc2ss::hash_combine(seed, R_src_c);
    return seed;
}

uint64_t Model_diode_bridge_3l::Switches::all() const {
    return 0 |
        (S_0_a << 0) |
        (S_0_b << 1) |
        (S_0_c << 2) |
        (S_D_n_a << 3) |
        (S_D_n_b << 4) |
        (S_D_n_c << 5) |
        (S_D_p_a << 6) |
        (S_D_p_b << 7) |
        (S_D_p_c << 8) |
        (S_n_a << 9) |
        (S_n_b << 10) |
        (S_n_c << 11) |
        (S_p_a << 12) |
        (S_p_b << 13) |
        (S_p_c << 14);
}

double Model_diode_bridge_3l::Switches::smallestDelay() {
    return std::min({double(rlc2ss::OnOffDelay::MAX_DELAY),
                    S_0_a.pendingTime(),
                    S_0_b.pendingTime(),
                    S_0_c.pendingTime(),
                    S_D_n_a.pendingTime(),
                    S_D_n_b.pendingTime(),
                    S_D_n_c.pendingTime(),
                    S_D_p_a.pendingTime(),
                    S_D_p_b.pendingTime(),
                    S_D_p_c.pendingTime(),
                    S_n_a.pendingTime(),
                    S_n_b.pendingTime(),
                    S_n_c.pendingTime(),
                    S_p_a.pendingTime(),
                    S_p_b.pendingTime(),
                    S_p_c.pendingTime()});
}

void Model_diode_bridge_3l::Switches::step(double dt) {
    S_0_a.step(dt);
    S_0_b.step(dt);
    S_0_c.step(dt);
    S_D_n_a.step(dt);
    S_D_n_b.step(dt);
    S_D_n_c.step(dt);
    S_D_p_a.step(dt);
    S_D_p_b.step(dt);
    S_D_p_c.step(dt);
    S_n_a.step(dt);
    S_n_b.step(dt);
    S_n_c.step(dt);
    S_p_a.step(dt);
    S_p_b.step(dt);
    S_p_c.step(dt);
}
