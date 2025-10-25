
#include "converter_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <fstream>
#include <format>

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

Model_converter::StateSpaceMatrices getStateSpaceMatrices(Model_converter::Components const& components, Model_converter::Switches const& switches);

static std::unique_ptr<Model_converter::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_converter::NUM_STATES, Model_converter::NUM_STATES> const& K1,
    Eigen::Matrix<double, Model_converter::NUM_STATES, Model_converter::NUM_STATES> const& A1,
    Eigen::Matrix<double, Model_converter::NUM_STATES, Model_converter::NUM_INPUTS> const& B1,
    Eigen::Matrix<double, Model_converter::NUM_OUTPUTS, Model_converter::NUM_STATES> const& K2,
    Eigen::Matrix<double, Model_converter::NUM_OUTPUTS, Model_converter::NUM_STATES> const& C1,
    Eigen::Matrix<double, Model_converter::NUM_OUTPUTS, Model_converter::NUM_INPUTS> const& D1) {
    auto ss = std::make_unique<Model_converter::StateSpaceMatrices>();
    ss->A = K1.partialPivLu().solve(A1);
    ss->B = K1.partialPivLu().solve(B1);
    ss->C = (C1 + K2 * ss->A);
    ss->D = (D1 + K2 * ss->B);
    return ss;
}

static std::optional<rlc2ss::ZeroCrossingEvent> checkZeroCrossingEvents(Model_converter& circuit, Model_converter::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    // Diode D_a_p
    double V_D_a_p = circuit.outputs.N_c_a - circuit.outputs.N_dc_p;
    if (V_D_a_p > circuit.inputs.V_D_a_p && !circuit.switches.S_D_a_p) {
        double V_D_a_p_prev = prev_outputs.N_c_a - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_a_p_prev, V_D_a_p),
            .event_callback = [&]() {
                circuit.switches.S_D_a_p = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_a_p < 0 && circuit.switches.S_D_a_p) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_a_p, circuit.outputs.I_R_D_a_p),
            .event_callback = [&]() {
                circuit.switches.S_D_a_p = 0;
            }
        });
    }

    // Diode D_b_p
    double V_D_b_p = circuit.outputs.N_c_b - circuit.outputs.N_dc_p;
    if (V_D_b_p > circuit.inputs.V_D_b_p && !circuit.switches.S_D_b_p) {
        double V_D_b_p_prev = prev_outputs.N_c_b - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_b_p_prev, V_D_b_p),
            .event_callback = [&]() {
                circuit.switches.S_D_b_p = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_b_p < 0 && circuit.switches.S_D_b_p) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_b_p, circuit.outputs.I_R_D_b_p),
            .event_callback = [&]() {
                circuit.switches.S_D_b_p = 0;
            }
        });
    }

    // Diode D_c_p
    double V_D_c_p = circuit.outputs.N_c_c - circuit.outputs.N_dc_p;
    if (V_D_c_p > circuit.inputs.V_D_c_p && !circuit.switches.S_D_c_p) {
        double V_D_c_p_prev = prev_outputs.N_c_c - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_c_p_prev, V_D_c_p),
            .event_callback = [&]() {
                circuit.switches.S_D_c_p = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_c_p < 0 && circuit.switches.S_D_c_p) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_c_p, circuit.outputs.I_R_D_c_p),
            .event_callback = [&]() {
                circuit.switches.S_D_c_p = 0;
            }
        });
    }

    // Diode D_a_n
    double V_D_a_n = circuit.outputs.N_dc_n - circuit.outputs.N_c_a;
    if (V_D_a_n > circuit.inputs.V_D_a_n && !circuit.switches.S_D_a_n) {
        double V_D_a_n_prev = prev_outputs.N_dc_n - prev_outputs.N_c_a;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_a_n_prev, V_D_a_n),
            .event_callback = [&]() {
                circuit.switches.S_D_a_n = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_a_n < 0 && circuit.switches.S_D_a_n) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_a_n, circuit.outputs.I_R_D_a_n),
            .event_callback = [&]() {
                circuit.switches.S_D_a_n = 0;
            }
        });
    }

    // Diode D_b_n
    double V_D_b_n = circuit.outputs.N_dc_n - circuit.outputs.N_c_b;
    if (V_D_b_n > circuit.inputs.V_D_b_n && !circuit.switches.S_D_b_n) {
        double V_D_b_n_prev = prev_outputs.N_dc_n - prev_outputs.N_c_b;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_b_n_prev, V_D_b_n),
            .event_callback = [&]() {
                circuit.switches.S_D_b_n = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_b_n < 0 && circuit.switches.S_D_b_n) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_b_n, circuit.outputs.I_R_D_b_n),
            .event_callback = [&]() {
                circuit.switches.S_D_b_n = 0;
            }
        });
    }

    // Diode D_c_n
    double V_D_c_n = circuit.outputs.N_dc_n - circuit.outputs.N_c_c;
    if (V_D_c_n > circuit.inputs.V_D_c_n && !circuit.switches.S_D_c_n) {
        double V_D_c_n_prev = prev_outputs.N_dc_n - prev_outputs.N_c_c;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_c_n_prev, V_D_c_n),
            .event_callback = [&]() {
                circuit.switches.S_D_c_n = 1;
            }
        });
    }
    if (circuit.outputs.I_R_D_c_n < 0 && circuit.switches.S_D_c_n) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_c_n, circuit.outputs.I_R_D_c_n),
            .event_callback = [&]() {
                circuit.switches.S_D_c_n = 0;
            }
        });
    }

    if (events.size() > 0) {
        return events.top();
    }
    return std::nullopt;
}

Model_converter::Model_converter(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
    m_ss = getStateSpaceMatrices(components, switches);
    m_solver.updateJacobian(m_ss.A);
}

void Model_converter::step(double dt, Inputs const& inputs_) {
    inputs.data = inputs_.data;

    // Copy previous state and outputs if step needs to be redone
    Model_converter::States prev_state;
    Model_converter::Outputs prev_outputs;
    prev_state.data = states.data;
    prev_outputs.data = outputs.data;

    stepInternal(dt);
    std::optional<rlc2ss::ZeroCrossingEvent> zc_event = checkZeroCrossingEvents(*this, prev_outputs);
    while (zc_event) {
        // Redo step
        states.data = prev_state.data;
        stepInternal(zc_event->time * dt);
        // Process event
        zc_event->event_callback();
        // Run remaining time
        prev_state.data = states.data;
        prev_outputs.data = outputs.data;
        dt = dt * (1 - zc_event->time);
        stepInternal(dt);
        // Check for new events
        zc_event = checkZeroCrossingEvents(*this, prev_outputs);
    }
}

void Model_converter::stepInternal(double dt) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all != _M_switches_DO_NOT_TOUCH.all || !m_solver.initialized()) {
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
        _M_switches_DO_NOT_TOUCH.all = switches.all;
        m_ss = getStateSpaceMatrices(components, switches);
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

struct Model_converter_Topology {
    Model_converter::Components components;
    Model_converter::Switches switches;
    std::unique_ptr<Model_converter::StateSpaceMatrices> state_space;
};

Model_converter::StateSpaceMatrices getStateSpaceMatrices(Model_converter::Components const& components, Model_converter::Switches const& switches) {
    static std::vector<Model_converter_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_converter_Topology const& t) {
            return t.components == components && t.switches.all == switches.all;
        });
    if (it != state_space_cache.end()) {
        return *it->state_space;
    }

    // The json file with symbolic intermediate matrices
    static nlohmann::json circuit_json;
    if (circuit_json.empty()) {
        circuit_json = nlohmann::json::parse(std::ifstream("c:\\Projects\\rlc2ss\\qucs\\converter_matrices.json"));
    }
    if (!circuit_json.contains(std::to_string(switches.all))) {
        circuit_json = nlohmann::json::parse(std::ifstream("c:\\Projects\\rlc2ss\\qucs\\converter_matrices.json"));
        if (!circuit_json.contains(std::to_string(switches.all))) {
            system(std::format("C:\\Projects\\rlc2ss\\.venv\\Scripts\\python.exe C:\\Projects\\rlc2ss\\scripts\\rlc2ss.py. c:\\Projects\\rlc2ss\\qucs\\converter.cir --combination={}", switches.all).c_str());
        }
        circuit_json = nlohmann::json::parse(std::ifstream("c:\\Projects\\rlc2ss\\qucs\\converter_matrices.json"));
    }
    
    assert(circuit_json.contains(std::to_string(switches.all)));

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = circuit_json[std::to_string(switches.all)].dump();
	s = rlc2ss::replace(s, "C_n", std::to_string(components.C_n));
	s = rlc2ss::replace(s, "C_p", std::to_string(components.C_p));
	s = rlc2ss::replace(s, "L_a", std::to_string(components.L_a));
	s = rlc2ss::replace(s, "L_b", std::to_string(components.L_b));
	s = rlc2ss::replace(s, "L_c", std::to_string(components.L_c));
	s = rlc2ss::replace(s, "R_D_a_n", std::to_string(components.R_D_a_n));
	s = rlc2ss::replace(s, "R_D_a_p", std::to_string(components.R_D_a_p));
	s = rlc2ss::replace(s, "R_D_b_n", std::to_string(components.R_D_b_n));
	s = rlc2ss::replace(s, "R_D_b_p", std::to_string(components.R_D_b_p));
	s = rlc2ss::replace(s, "R_D_c_n", std::to_string(components.R_D_c_n));
	s = rlc2ss::replace(s, "R_D_c_p", std::to_string(components.R_D_c_p));
	s = rlc2ss::replace(s, "R_a", std::to_string(components.R_a));
	s = rlc2ss::replace(s, "R_b", std::to_string(components.R_b));
	s = rlc2ss::replace(s, "R_c", std::to_string(components.R_c));
	s = rlc2ss::replace(s, "R_dc", std::to_string(components.R_dc));
	s = rlc2ss::replace(s, "R_n_p", std::to_string(components.R_n_p));
	s = rlc2ss::replace(s, "R_n_s", std::to_string(components.R_n_s));
	s = rlc2ss::replace(s, "R_p_p", std::to_string(components.R_p_p));
	s = rlc2ss::replace(s, "R_p_s", std::to_string(components.R_p_s));

    // Parse json for the intermediate matrices
    nlohmann::json j = nlohmann::json::parse(s);
    std::string K1_str = j["K1"];
    std::string K2_str = j["K2"];
    std::string A1_str = j["A1"];
    std::string B1_str = j["B1"];
    std::string C1_str = j["C1"];
    std::string D1_str = j["D1"];

    // Create eigen matrices
    Eigen::Matrix<double, Model_converter::NUM_STATES, Model_converter::NUM_STATES, Eigen::RowMajor> K1(rlc2ss::getCommaDelimitedValues(K1_str).data());
    Eigen::Matrix<double, Model_converter::NUM_OUTPUTS, Model_converter::NUM_STATES, Eigen::RowMajor> K2(rlc2ss::getCommaDelimitedValues(K2_str).data());
    Eigen::Matrix<double, Model_converter::NUM_STATES, Model_converter::NUM_STATES, Eigen::RowMajor> A1(rlc2ss::getCommaDelimitedValues(A1_str).data());
    Eigen::Matrix<double, Model_converter::NUM_STATES, Model_converter::NUM_INPUTS, Eigen::RowMajor> B1(rlc2ss::getCommaDelimitedValues(B1_str).data());
    Eigen::Matrix<double, Model_converter::NUM_OUTPUTS, Model_converter::NUM_STATES, Eigen::RowMajor> C1(rlc2ss::getCommaDelimitedValues(C1_str).data());
    Eigen::Matrix<double, Model_converter::NUM_OUTPUTS, Model_converter::NUM_INPUTS, Eigen::RowMajor> D1(rlc2ss::getCommaDelimitedValues(D1_str).data());

    Model_converter_Topology& topology = state_space_cache.emplace_back(Model_converter_Topology{
        .components = components,
        .switches = switches,
        .state_space = calcStateSpace(K1, A1, B1, K2, C1, D1)});

    return *topology.state_space;
}
