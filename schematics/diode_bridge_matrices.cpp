
#include "diode_bridge_matrices.hpp"
#include "rlc2ss.h"

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

Model_diode_bridge::Model_diode_bridge(Components const& c)
    : components(c),
      m_components_DO_NOT_TOUCH(c) {
    updateStateSpaceMatrices();
    m_solver.updateJacobian(m_ss.A);
}

void Model_diode_bridge::step(double dt, Inputs const& inputs_) {
    inputs.data = inputs_.data;
    // Update state-space matrices if needed
    if (components != m_components_DO_NOT_TOUCH || switches.all != m_switches_DO_NOT_TOUCH.all) {
		assert(components.L_a != -1);
		assert(components.L_b != -1);
		assert(components.L_c != -1);
		assert(components.C_dc != -1);
		assert(components.R_a != -1);
		assert(components.R_b != -1);
		assert(components.R_c != -1);
		assert(components.R_dc != -1);
		assert(components.R_load != -1);
        m_components_DO_NOT_TOUCH = components;
        m_switches_DO_NOT_TOUCH.all = switches.all;
        updateStateSpaceMatrices();
        m_solver.updateJacobian(m_ss.A);
        // Solve one step with backward euler to reduce numerical oscillations
        m_Bu = m_ss.B * inputs.data;
        states.data = m_solver.stepBackwardEuler(*this, states.data, 0.0, dt);
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
    m_dt_prev = dt;

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
            return t.components == components && t.switches.all == switches.all;
        });
    if (it != state_space_cache.end()) {
        m_ss = *it->state_space;
        return;
    }

    if (m_circuit_json.empty()) {
        m_circuit_json = nlohmann::json::parse(rlc2ss::loadTextResource(101));
    }
    assert(!m_circuit_json.empty());

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = m_circuit_json[std::to_string(switches.all)].dump();
	s = rlc2ss::replace(s, "L_a", std::to_string(components.L_a));
	s = rlc2ss::replace(s, "L_b", std::to_string(components.L_b));
	s = rlc2ss::replace(s, "L_c", std::to_string(components.L_c));
	s = rlc2ss::replace(s, "C_dc", std::to_string(components.C_dc));
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
