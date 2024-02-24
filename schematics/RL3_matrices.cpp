
#include "RL3_matrices.hpp"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<Model_RL3::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> const  &K1,
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> const  &A1,
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_INPUTS> const  &B1,
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> const &K2,
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> const &C1,
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_INPUTS> const &D1) {
    auto ss = std::make_unique<Model_RL3::StateSpaceMatrices>();
    ss->A   = K1.partialPivLu().solve(A1);
    ss->B   = K1.partialPivLu().solve(B1);
    ss->C   = (C1 + K2 * ss->A);
    ss->D   = (D1 + K2 * ss->B);
    return ss;
}

Model_RL3::Model_RL3(Components const& c)
    : components(c),
      m_components_DO_NOT_TOUCH(c) {
    m_ss = calculateStateSpace(components, switches);
    m_solver.updateJacobian(m_ss.A);
}


void Model_RL3::step(double dt, Inputs const& inputs_) {
    inputs.data = inputs_.data;
    // Update state-space matrices if needed
    if (components != m_components_DO_NOT_TOUCH || switches.all != m_switches_DO_NOT_TOUCH.all) {
		assert(components.L_a != -1);
		assert(components.L_b != -1);
		assert(components.L_c != -1);
		assert(components.R_a != -1);
		assert(components.R_b != -1);
		assert(components.R_c != -1);
		assert(components.Kab != -1);
		assert(components.Kbc != -1);
		assert(components.Kca != -1);
        m_components_DO_NOT_TOUCH = components;
        m_switches_DO_NOT_TOUCH.all = switches.all;
        m_ss = calculateStateSpace(components, switches);
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
}
std::unique_ptr<Model_RL3::StateSpaceMatrices> calculateStateSpace_0(Model_RL3::Components const& c);

struct Model_RL3_Topology {
    Model_RL3::Components components;
    Model_RL3::Switches switches;
    std::unique_ptr<Model_RL3::StateSpaceMatrices> state_space;
};

Model_RL3::StateSpaceMatrices Model_RL3::calculateStateSpace(Model_RL3::Components const& components, Model_RL3::Switches switches)
{
    static std::vector<Model_RL3_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_RL3_Topology const& t) {
        return t.components == components && t.switches.all == switches.all;
    });
    if (it != state_space_cache.end()) {
        return *it->state_space;
    }
    auto state_space = std::make_unique<Model_RL3::StateSpaceMatrices>();

    switch (switches.all) {
		case 0: state_space = calculateStateSpace_0(components); break;
    default:
        assert(("Invalid switch combination", 0));
    }
    Model_RL3_Topology& topology = state_space_cache.emplace_back(Model_RL3_Topology{
        .components = components,
        .switches = switches,
        .state_space = std::move(state_space)});

    return *topology.state_space;
}

std::unique_ptr<Model_RL3::StateSpaceMatrices> calculateStateSpace_0(Model_RL3::Components const& c) // 
{
	double L_a = c.L_a;
	double L_b = c.L_b;
	double L_c = c.L_c;
	double R_a = c.R_a;
	double R_b = c.R_b;
	double R_c = c.R_c;
	double Kab = c.Kab;
	double Kbc = c.Kbc;
	double Kca = c.Kca;


    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> K1 {
		{ 0, -2*Kab*sqrt(L_a*L_b) + L_a + L_b, -Kab*sqrt(L_a*L_b) + Kbc*sqrt(L_b*L_c) - Kca*sqrt(L_a*L_c) + L_a },
		{ 0, -Kab*sqrt(L_a*L_b) + Kbc*sqrt(L_b*L_c) - Kca*sqrt(L_a*L_c) + L_a, -2*Kca*sqrt(L_a*L_c) + L_a + L_c },
		{ 1, 1, 1 } };

    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> K2 {
		{ 0, 0, 0 },
		{ 0, 0, 0 },
		{ 0, 0, 0} };

    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> A1 {
		{ 0, -R_a - R_b, -R_a },
		{ 0, -R_a, -R_a - R_c },
		{ 0, 0, 0 } };

    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_INPUTS> B1 {
		{ -1, 1, 0 },
		{ -1, 0, 1 },
		{ 0, 0, 0 } };

    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> C1 {
		{ 0, -1, -1 },
		{ 0, 1, 0 },
		{ 0, 0, 1 } };

    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_INPUTS> D1 {
		{ 0, 0, 0 },
		{ 0, 0, 0 },
		{ 0, 0, 0 } };

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

