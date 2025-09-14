
#include "controlled_sources_matrices.hpp"
#include "rlc2ss.h"
#include <optional>

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<Model_controlled_sources::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_controlled_sources::NUM_STATES, Model_controlled_sources::NUM_STATES> const  &K1,
    Eigen::Matrix<double, Model_controlled_sources::NUM_STATES, Model_controlled_sources::NUM_STATES> const  &A1,
    Eigen::Matrix<double, Model_controlled_sources::NUM_STATES, Model_controlled_sources::NUM_INPUTS> const  &B1,
    Eigen::Matrix<double, Model_controlled_sources::NUM_OUTPUTS, Model_controlled_sources::NUM_STATES> const &K2,
    Eigen::Matrix<double, Model_controlled_sources::NUM_OUTPUTS, Model_controlled_sources::NUM_STATES> const &C1,
    Eigen::Matrix<double, Model_controlled_sources::NUM_OUTPUTS, Model_controlled_sources::NUM_INPUTS> const &D1) {
    auto ss = std::make_unique<Model_controlled_sources::StateSpaceMatrices>();
    ss->A = K1.partialPivLu().solve(A1);
    ss->B = K1.partialPivLu().solve(B1);
    ss->C = (C1 + K2 * ss->A);
    ss->D = (D1 + K2 * ss->B);
    return ss;
}

static std::optional<rlc2ss::ZeroCrossingEvent> checkZeroCrossingEvents(Model_controlled_sources& circuit, Model_controlled_sources::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    if (events.size() > 0) {
        return events.top();
    }
    return std::nullopt;
}

Model_controlled_sources::Model_controlled_sources(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
    m_ss = calculateStateSpace(components, switches);
    m_solver.updateJacobian(m_ss.A);
}


void Model_controlled_sources::step(double dt, Inputs const& inputs_) {
    inputs.data = inputs_.data;

    // Copy previous state and outputs if step needs to be redone
    Model_controlled_sources::States prev_state;
    Model_controlled_sources::Outputs prev_outputs;
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

void Model_controlled_sources::stepInternal(double dt) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all != _M_switches_DO_NOT_TOUCH.all) {
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
        _M_components_DO_NOT_TOUCH = components;
        _M_switches_DO_NOT_TOUCH.all = switches.all;
        m_ss = calculateStateSpace(components, switches);
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
	states.I_L1 = outputs.I_L1;
	states.V_C_1 = outputs.V_C_1;
	states.V_C_2 = outputs.V_C_2;
}
std::unique_ptr<Model_controlled_sources::StateSpaceMatrices> calculateStateSpace_0(Model_controlled_sources::Components const& c);

struct Model_controlled_sources_Topology {
    Model_controlled_sources::Components components;
    Model_controlled_sources::Switches switches;
    std::unique_ptr<Model_controlled_sources::StateSpaceMatrices> state_space;
};

Model_controlled_sources::StateSpaceMatrices Model_controlled_sources::calculateStateSpace(Model_controlled_sources::Components const& components, Model_controlled_sources::Switches switches)
{
    static std::vector<Model_controlled_sources_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_controlled_sources_Topology const& t) {
        return t.components == components && t.switches.all == switches.all;
    });
    if (it != state_space_cache.end()) {
        return *it->state_space;
    }
    auto state_space = std::make_unique<Model_controlled_sources::StateSpaceMatrices>();

    switch (switches.all) {
		case 0: state_space = calculateStateSpace_0(components); break;
    default:
        assert(("Invalid switch combination", 0));
    }
    Model_controlled_sources_Topology& topology = state_space_cache.emplace_back(Model_controlled_sources_Topology{
        .components = components,
        .switches = switches,
        .state_space = std::move(state_space)});

    return *topology.state_space;
}

std::unique_ptr<Model_controlled_sources::StateSpaceMatrices> calculateStateSpace_0(Model_controlled_sources::Components const& c) // 
{
	double C_1 = c.C_1;
	double C_2 = c.C_2;
	double ESRC3 = c.ESRC3;
	double FSRC5 = c.FSRC5;
	double GSRC1 = c.GSRC1;
	double HSRC4 = c.HSRC4;
	double L1 = c.L1;
	double R1 = c.R1;
	double R2 = c.R2;
	double R3 = c.R3;
	double R4 = c.R4;
	double R5 = c.R5;


    Eigen::Matrix<double, Model_controlled_sources::NUM_STATES, Model_controlled_sources::NUM_STATES> K1 {
		{ -FSRC5*GSRC1*L1*R3 + L1, 0, 0 },
		{ GSRC1*L1, C_1, 0 },
		{ -FSRC5*GSRC1*L1, 0, C_2 } };

    Eigen::Matrix<double, Model_controlled_sources::NUM_OUTPUTS, Model_controlled_sources::NUM_STATES> K2 {
		{ 0, 0, 0 },
		{ 0, 0, 0 },
		{ 0, 0, 0} };

    Eigen::Matrix<double, Model_controlled_sources::NUM_STATES, Model_controlled_sources::NUM_STATES> A1 {
		{ HSRC4 - R1 - R3 - R5, ESRC3, 0 },
		{ 0, 0, 0 },
		{ 0, 0, 0 } };

    Eigen::Matrix<double, Model_controlled_sources::NUM_STATES, Model_controlled_sources::NUM_INPUTS> B1 {
		{ 1, -1, 0 },
		{ 0, 0, 0 },
		{ 0, 0, 0 } };

    Eigen::Matrix<double, Model_controlled_sources::NUM_OUTPUTS, Model_controlled_sources::NUM_STATES> C1 {
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 0, 0, 1 } };

    Eigen::Matrix<double, Model_controlled_sources::NUM_OUTPUTS, Model_controlled_sources::NUM_INPUTS> D1 {
		{ 0, 0, 0 },
		{ 0, 0, 0 },
		{ 0, 0, 0 } };

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

