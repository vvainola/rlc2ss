
#include "diode_bridge_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <fstream>
#include <format>
#include <memory>
#include "diode_bridge_matrices_json.h"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<Model_diode_bridge::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_diode_bridge::NUM_STATES, Model_diode_bridge::NUM_STATES> const& K1,
    Eigen::Matrix<double, Model_diode_bridge::NUM_STATES, Model_diode_bridge::NUM_STATES> const& A1,
    Eigen::Matrix<double, Model_diode_bridge::NUM_STATES, Model_diode_bridge::NUM_INPUTS> const& B1,
    Eigen::Matrix<double, Model_diode_bridge::NUM_OUTPUTS, Model_diode_bridge::NUM_STATES> const& K2,
    Eigen::Matrix<double, Model_diode_bridge::NUM_OUTPUTS, Model_diode_bridge::NUM_STATES> const& C1,
    Eigen::Matrix<double, Model_diode_bridge::NUM_OUTPUTS, Model_diode_bridge::NUM_INPUTS> const& D1) {
    auto ss = std::make_unique<Model_diode_bridge::StateSpaceMatrices>();
    ss->A = K1.partialPivLu().solve(A1);
    ss->B = K1.partialPivLu().solve(B1);
    ss->C = (C1 + K2 * ss->A);
    ss->D = (D1 + K2 * ss->B);
    return ss;
}

static std::optional<rlc2ss::ZeroCrossingEvent> checkZeroCrossingEvents(Model_diode_bridge& circuit, Model_diode_bridge::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    // Diode D_p_a
    double V_D_p_a = circuit.outputs._net0 - circuit.outputs.N_dc_p;
    if (V_D_p_a > circuit.inputs.V_D_p_a && !circuit.switches.S_D_p_a) {
        double V_D_p_a_prev = prev_outputs._net0 - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_p_a_prev, V_D_p_a),
            .event_callback = [&]() {
                circuit.switches.S_D_p_a = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_p_a < 0 && circuit.switches.S_D_p_a) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_p_a, circuit.outputs.I_R_D_p_a),
            .event_callback = [&]() {
                circuit.switches.S_D_p_a = 0;
            }
        });
    }

    // Diode D_p_b
    double V_D_p_b = circuit.outputs._net1 - circuit.outputs.N_dc_p;
    if (V_D_p_b > circuit.inputs.V_D_p_b && !circuit.switches.S_D_p_b) {
        double V_D_p_b_prev = prev_outputs._net1 - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_p_b_prev, V_D_p_b),
            .event_callback = [&]() {
                circuit.switches.S_D_p_b = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_p_b < 0 && circuit.switches.S_D_p_b) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_p_b, circuit.outputs.I_R_D_p_b),
            .event_callback = [&]() {
                circuit.switches.S_D_p_b = 0;
            }
        });
    }

    // Diode D_p_c
    double V_D_p_c = circuit.outputs._net2 - circuit.outputs.N_dc_p;
    if (V_D_p_c > circuit.inputs.V_D_p_c && !circuit.switches.S_D_p_c) {
        double V_D_p_c_prev = prev_outputs._net2 - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_p_c_prev, V_D_p_c),
            .event_callback = [&]() {
                circuit.switches.S_D_p_c = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_p_c < 0 && circuit.switches.S_D_p_c) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_p_c, circuit.outputs.I_R_D_p_c),
            .event_callback = [&]() {
                circuit.switches.S_D_p_c = 0;
            }
        });
    }

    // Diode D_n_a
    double V_D_n_a = circuit.outputs.N_dc_n - circuit.outputs._net0;
    if (V_D_n_a > circuit.inputs.V_D_n_a && !circuit.switches.S_D_n_a) {
        double V_D_n_a_prev = prev_outputs.N_dc_n - prev_outputs._net0;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_n_a_prev, V_D_n_a),
            .event_callback = [&]() {
                circuit.switches.S_D_n_a = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_n_a < 0 && circuit.switches.S_D_n_a) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_n_a, circuit.outputs.I_R_D_n_a),
            .event_callback = [&]() {
                circuit.switches.S_D_n_a = 0;
            }
        });
    }

    // Diode D_n_b
    double V_D_n_b = circuit.outputs.N_dc_n - circuit.outputs._net1;
    if (V_D_n_b > circuit.inputs.V_D_n_b && !circuit.switches.S_D_n_b) {
        double V_D_n_b_prev = prev_outputs.N_dc_n - prev_outputs._net1;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_n_b_prev, V_D_n_b),
            .event_callback = [&]() {
                circuit.switches.S_D_n_b = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_n_b < 0 && circuit.switches.S_D_n_b) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_n_b, circuit.outputs.I_R_D_n_b),
            .event_callback = [&]() {
                circuit.switches.S_D_n_b = 0;
            }
        });
    }

    // Diode D_n_c
    double V_D_n_c = circuit.outputs.N_dc_n - circuit.outputs._net2;
    if (V_D_n_c > circuit.inputs.V_D_n_c && !circuit.switches.S_D_n_c) {
        double V_D_n_c_prev = prev_outputs.N_dc_n - prev_outputs._net2;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_n_c_prev, V_D_n_c),
            .event_callback = [&]() {
                circuit.switches.S_D_n_c = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_n_c < 0 && circuit.switches.S_D_n_c) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_n_c, circuit.outputs.I_R_D_n_c),
            .event_callback = [&]() {
                circuit.switches.S_D_n_c = 0;
            }
        });
    }

    if (events.size() > 0) {
        return events.top();
    }
    return std::nullopt;
}

Model_diode_bridge::Model_diode_bridge(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
    updateStateSpaceMatrices();
    m_solver.updateJacobian(m_ss.A);
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
    std::optional<rlc2ss::ZeroCrossingEvent> zc_event = checkZeroCrossingEvents(*this, prev_outputs);
    while (zc_event) {
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
        zc_event = checkZeroCrossingEvents(*this, prev_outputs);
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
    states.V_C_dc = outputs.V_C_dc;
}

struct Model_diode_bridge_Topology {
    Model_diode_bridge::Components components;
    Model_diode_bridge::Switches switches;
    std::unique_ptr<Model_diode_bridge::StateSpaceMatrices> state_space;
};

void Model_diode_bridge::updateStateSpaceMatrices() {
    static std::vector<Model_diode_bridge_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_diode_bridge_Topology const& t) {
            return t.components == components && t.switches.all() == switches.all();
        });
    if (it != state_space_cache.end()) {
        m_ss = *it->state_space;
        return;
    }

    if (m_circuit_json.empty()) {
        m_circuit_json = nlohmann::json::parse(std::string(diode_bridge_matrices_json_hexdump, diode_bridge_matrices_json_hexdump + diode_bridge_matrices_json_hexdump_len));
    }
    assert(m_circuit_json.contains(std::to_string(switches.all())));

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = m_circuit_json[std::to_string(switches.all())].dump();
	s = rlc2ss::replace(s, "C_dc", std::to_string(components.C_dc));
	s = rlc2ss::replace(s, "L_a", std::to_string(components.L_a));
	s = rlc2ss::replace(s, "L_b", std::to_string(components.L_b));
	s = rlc2ss::replace(s, "L_c", std::to_string(components.L_c));
	s = rlc2ss::replace(s, "R_D_n_a", std::to_string(components.R_D_n_a));
	s = rlc2ss::replace(s, "R_D_n_b", std::to_string(components.R_D_n_b));
	s = rlc2ss::replace(s, "R_D_n_c", std::to_string(components.R_D_n_c));
	s = rlc2ss::replace(s, "R_D_p_a", std::to_string(components.R_D_p_a));
	s = rlc2ss::replace(s, "R_D_p_b", std::to_string(components.R_D_p_b));
	s = rlc2ss::replace(s, "R_D_p_c", std::to_string(components.R_D_p_c));
	s = rlc2ss::replace(s, "R_a", std::to_string(components.R_a));
	s = rlc2ss::replace(s, "R_b", std::to_string(components.R_b));
	s = rlc2ss::replace(s, "R_c", std::to_string(components.R_c));
	s = rlc2ss::replace(s, "R_dc", std::to_string(components.R_dc));
	s = rlc2ss::replace(s, "R_load", std::to_string(components.R_load));

    // Parse json for the intermediate matrices
    nlohmann::json j = nlohmann::json::parse(s);
    std::string K1_str = j["K1"];
    std::string K2_str = j["K2"];
    std::string A1_str = j["A1"];
    std::string B1_str = j["B1"];
    std::string C1_str = j["C1"];
    std::string D1_str = j["D1"];

    // Create eigen matrices
    Eigen::Matrix<double, Model_diode_bridge::NUM_STATES, Model_diode_bridge::NUM_STATES, Eigen::RowMajor> K1(rlc2ss::getCommaDelimitedValues(K1_str).data());
    Eigen::Matrix<double, Model_diode_bridge::NUM_OUTPUTS, Model_diode_bridge::NUM_STATES, Eigen::RowMajor> K2(rlc2ss::getCommaDelimitedValues(K2_str).data());
    Eigen::Matrix<double, Model_diode_bridge::NUM_STATES, Model_diode_bridge::NUM_STATES, Eigen::RowMajor> A1(rlc2ss::getCommaDelimitedValues(A1_str).data());
    Eigen::Matrix<double, Model_diode_bridge::NUM_STATES, Model_diode_bridge::NUM_INPUTS, Eigen::RowMajor> B1(rlc2ss::getCommaDelimitedValues(B1_str).data());
    Eigen::Matrix<double, Model_diode_bridge::NUM_OUTPUTS, Model_diode_bridge::NUM_STATES, Eigen::RowMajor> C1(rlc2ss::getCommaDelimitedValues(C1_str).data());
    Eigen::Matrix<double, Model_diode_bridge::NUM_OUTPUTS, Model_diode_bridge::NUM_INPUTS, Eigen::RowMajor> D1(rlc2ss::getCommaDelimitedValues(D1_str).data());

    Model_diode_bridge_Topology& topology = state_space_cache.emplace_back(Model_diode_bridge_Topology{
        .components = components,
        .switches = switches,
        .state_space = calcStateSpace(K1, A1, B1, K2, C1, D1)});

    m_ss = *topology.state_space;
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
