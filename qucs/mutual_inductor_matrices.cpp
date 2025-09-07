
#include "mutual_inductor_matrices.hpp"
#include "rlc2ss.h"
#include <optional>

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_mutual_inductor::NUM_STATES, Model_mutual_inductor::NUM_STATES> const  &K1,
    Eigen::Matrix<double, Model_mutual_inductor::NUM_STATES, Model_mutual_inductor::NUM_STATES> const  &A1,
    Eigen::Matrix<double, Model_mutual_inductor::NUM_STATES, Model_mutual_inductor::NUM_INPUTS> const  &B1,
    Eigen::Matrix<double, Model_mutual_inductor::NUM_OUTPUTS, Model_mutual_inductor::NUM_STATES> const &K2,
    Eigen::Matrix<double, Model_mutual_inductor::NUM_OUTPUTS, Model_mutual_inductor::NUM_STATES> const &C1,
    Eigen::Matrix<double, Model_mutual_inductor::NUM_OUTPUTS, Model_mutual_inductor::NUM_INPUTS> const &D1) {
    auto ss = std::make_unique<Model_mutual_inductor::StateSpaceMatrices>();
    ss->A = K1.partialPivLu().solve(A1);
    ss->B = K1.partialPivLu().solve(B1);
    ss->C = (C1 + K2 * ss->A);
    ss->D = (D1 + K2 * ss->B);
    return ss;
}

static std::optional<rlc2ss::ZeroCrossingEvent> checkZeroCrossingEvents(Model_mutual_inductor& circuit, Model_mutual_inductor::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    if (events.size() > 0) {
        return events.top();
    }
    return std::nullopt;
}

Model_mutual_inductor::Model_mutual_inductor(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
    m_ss = calculateStateSpace(components, switches);
    m_solver.updateJacobian(m_ss.A);
}


void Model_mutual_inductor::step(double dt, Inputs const& inputs_) {
    inputs.data = inputs_.data;

    // Copy previous state and outputs if step needs to be redone
    Model_mutual_inductor::States prev_state;
    Model_mutual_inductor::Outputs prev_outputs;
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

void Model_mutual_inductor::stepInternal(double dt) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all != _M_switches_DO_NOT_TOUCH.all) {
		assert(components.K12 != -1);
		assert(components.K21 != -1);
		assert(components.K31 != -1);
		assert(components.L1 != -1);
		assert(components.L2 != -1);
		assert(components.L3 != -1);
		assert(components.R1 != -1);
		assert(components.R2 != -1);
		assert(components.R3 != -1);
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
	states.I_L2 = outputs.I_L2;
	states.I_L3 = outputs.I_L3;
}
std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices> calculateStateSpace_0(Model_mutual_inductor::Components const& c);

struct Model_mutual_inductor_Topology {
    Model_mutual_inductor::Components components;
    Model_mutual_inductor::Switches switches;
    std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices> state_space;
};

Model_mutual_inductor::StateSpaceMatrices Model_mutual_inductor::calculateStateSpace(Model_mutual_inductor::Components const& components, Model_mutual_inductor::Switches switches)
{
    static std::vector<Model_mutual_inductor_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_mutual_inductor_Topology const& t) {
        return t.components == components && t.switches.all == switches.all;
    });
    if (it != state_space_cache.end()) {
        return *it->state_space;
    }
    auto state_space = std::make_unique<Model_mutual_inductor::StateSpaceMatrices>();

    switch (switches.all) {
		case 0: state_space = calculateStateSpace_0(components); break;
    default:
        assert(("Invalid switch combination", 0));
    }
    Model_mutual_inductor_Topology& topology = state_space_cache.emplace_back(Model_mutual_inductor_Topology{
        .components = components,
        .switches = switches,
        .state_space = std::move(state_space)});

    return *topology.state_space;
}

std::unique_ptr<Model_mutual_inductor::StateSpaceMatrices> calculateStateSpace_0(Model_mutual_inductor::Components const& c) // 
{
	double K12 = c.K12;
	double K21 = c.K21;
	double K31 = c.K31;
	double L1 = c.L1;
	double L2 = c.L2;
	double L3 = c.L3;
	double R1 = c.R1;
	double R2 = c.R2;
	double R3 = c.R3;


    Eigen::Matrix<double, Model_mutual_inductor::NUM_STATES, Model_mutual_inductor::NUM_STATES> K1 {
		{ L1, K12*sqrt(L1*L2), K31*sqrt(L1*L3) },
		{ K12*sqrt(L1*L2), L2, K21*sqrt(L2*L3) },
		{ K31*sqrt(L1*L3), K21*sqrt(L2*L3), L3 } };

    Eigen::Matrix<double, Model_mutual_inductor::NUM_OUTPUTS, Model_mutual_inductor::NUM_STATES> K2 {
		{ 0, 0, 0 },
		{ 0, 0, 0 },
		{ 0, 0, 0} };

    Eigen::Matrix<double, Model_mutual_inductor::NUM_STATES, Model_mutual_inductor::NUM_STATES> A1 {
		{ -R1, 0, 0 },
		{ 0, -R2, 0 },
		{ 0, 0, -R3 } };

    Eigen::Matrix<double, Model_mutual_inductor::NUM_STATES, Model_mutual_inductor::NUM_INPUTS> B1 {
		{ 1, 0, 0, -1 },
		{ 0, 1, 0, -1 },
		{ 0, 0, 1, -1 } };

    Eigen::Matrix<double, Model_mutual_inductor::NUM_OUTPUTS, Model_mutual_inductor::NUM_STATES> C1 {
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 0, 0, 1 } };

    Eigen::Matrix<double, Model_mutual_inductor::NUM_OUTPUTS, Model_mutual_inductor::NUM_INPUTS> D1 {
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 0 } };

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

