
#include "diode_matrices.hpp"
#include "rlc2ss.h"
#include <optional>

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<Model_diode::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> const& K1,
    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> const& A1,
    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_INPUTS> const& B1,
    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> const& K2,
    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> const& C1,
    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_INPUTS> const& D1) {
    auto ss = std::make_unique<Model_diode::StateSpaceMatrices>();
    ss->A = K1.partialPivLu().solve(A1);
    ss->B = K1.partialPivLu().solve(B1);
    ss->C = (C1 + K2 * ss->A);
    ss->D = (D1 + K2 * ss->B);
    return ss;
}

static std::optional<rlc2ss::ZeroCrossingEvent> checkZeroCrossingEvents(Model_diode& circuit, Model_diode::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    double V_D2 = circuit.outputs.N_D2_pos - circuit.outputs.N_D2_neg;
    if (V_D2 > circuit.inputs.V_D2 && !circuit.switches.S_D2) {
        double V_D2_prev = prev_outputs.N_D2_pos - prev_outputs.N_D2_neg;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D2_prev, V_D2),
            .event_callback = [&]() {
                circuit.switches.S_D2 = 1;
            }});
    }
    if (circuit.outputs.I_R_D2 < 0 && circuit.switches.S_D2) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D2, circuit.outputs.I_R_D2),
            .event_callback = [&]() {
                circuit.switches.S_D2 = 0;
            }});
    }

    double V_D3 = circuit.outputs.N_D3_pos - circuit.outputs.N_D3_neg;
    if (V_D3 > circuit.inputs.V_D3 && !circuit.switches.S_D3) {
        double V_D3_prev = prev_outputs.N_D3_pos - prev_outputs.N_D3_neg;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D3_prev, V_D3),
            .event_callback = [&]() {
                circuit.switches.S_D3 = 1;
            }});
    }
    if (circuit.outputs.I_R_D3 < 0 && circuit.switches.S_D3) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D3, circuit.outputs.I_R_D3),
            .event_callback = [&]() {
                circuit.switches.S_D3 = 0;
            }});
    }

    if (events.size() > 0) {
        return events.top();
    }
    return std::nullopt;
}

Model_diode::Model_diode(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
    m_ss = calculateStateSpace(components, switches);
    m_solver.updateJacobian(m_ss.A);
}

void Model_diode::step(double dt, Inputs const& inputs_) {
    inputs.data = inputs_.data;

    // Copy previous state and outputs if step needs to be redone
    Model_diode::States prev_state;
    Model_diode::Outputs prev_outputs;
    prev_state.data = states.data;
    prev_outputs.data = outputs.data;

    stepInternal(dt, true);
    std::optional<rlc2ss::ZeroCrossingEvent> zc_event = checkZeroCrossingEvents(*this, prev_outputs);
    while (zc_event) {
        // Redo step
        states.data = prev_state.data;
        stepInternal(zc_event->time * dt, false);
        // Process event
        zc_event->event_callback();
        // Run remaining time
        prev_state.data = states.data;
        prev_outputs.data = outputs.data;
        dt = dt * (1 - zc_event->time);
        stepInternal(dt, false);
        // Check for new events
        zc_event = checkZeroCrossingEvents(*this, prev_outputs);
    }
}

void Model_diode::stepInternal(double dt, bool check_topology) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all != _M_switches_DO_NOT_TOUCH.all) {
        assert(components.L1 != -1);
        assert(components.L2 != -1);
        assert(components.R1 != -1);
        assert(components.R2 != -1);
        assert(components.R3 != -1);
        assert(components.R4 != -1);
        assert(components.R_D2 != -1);
        assert(components.R_D3 != -1);
        m_ss = calculateStateSpace(components, switches);
        if (check_topology) {
            checkTopology();
        }
        _M_components_DO_NOT_TOUCH = components;
        _M_switches_DO_NOT_TOUCH.all = switches.all;
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
}

void Model_diode::checkTopology() {
    // Update output
    Model_diode::Inputs inputs2;
    inputs2.data = inputs.data;
    inputs2.V_internal = outputs.V_L1;
    Model_diode::States states2;
    states2.data = states.data;
    states2.I_L1 = 0;
    Model_diode::Outputs outputs2;
    // Check if the topology is invalid i.e. inductor current is forced to zero
    outputs2.data = m_ss.C * states2.data + m_ss.D * inputs2.data;
    bool invalid_topology = outputs2.I_L1 != states.I_L1
                         || outputs2.I_L2 != states.I_L2;

    // 
    double V_D2 = outputs2.N_D2_pos - outputs2.N_D2_neg;
    if (V_D2 > inputs.V_D2 && !switches.S_D2) {
        switches.S_D2 = 1;
        m_ss = calculateStateSpace(components, switches);
        checkTopology();
    }
    double V_D3 = outputs2.N_D3_pos - outputs2.N_D3_neg;
    if (V_D3 > inputs.V_D3 && !switches.S_D3) {
        switches.S_D3 = 1;
        m_ss = calculateStateSpace(components, switches);
        checkTopology();
    }

    if (invalid_topology) {
        // Turn on the diode with highest voltage
        if (V_D2 > V_D3) {
            switches.S_D2 = 1;
        } else {
            switches.S_D3 = 1;
        }
        m_ss = calculateStateSpace(components, switches);
    }
}
std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_0(Model_diode::Components const& c);
std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_1(Model_diode::Components const& c);
std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_2(Model_diode::Components const& c);
std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_3(Model_diode::Components const& c);
std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_4(Model_diode::Components const& c);
std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_5(Model_diode::Components const& c);
std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_6(Model_diode::Components const& c);
std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_7(Model_diode::Components const& c);

struct Model_diode_Topology {
    Model_diode::Components components;
    Model_diode::Switches switches;
    std::unique_ptr<Model_diode::StateSpaceMatrices> state_space;
};

Model_diode::StateSpaceMatrices Model_diode::calculateStateSpace(Model_diode::Components const& components, Model_diode::Switches switches) {
    static std::vector<Model_diode_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_diode_Topology const& t) {
            return t.components == components && t.switches.all == switches.all;
        });
    if (it != state_space_cache.end()) {
        return *it->state_space;
    }
    auto state_space = std::make_unique<Model_diode::StateSpaceMatrices>();

    switch (switches.all) {
        case 0: state_space = calculateStateSpace_0(components); break;
        case 1: state_space = calculateStateSpace_1(components); break;
        case 2: state_space = calculateStateSpace_2(components); break;
        case 3: state_space = calculateStateSpace_3(components); break;
        case 4: state_space = calculateStateSpace_4(components); break;
        case 5: state_space = calculateStateSpace_5(components); break;
        case 6: state_space = calculateStateSpace_6(components); break;
        case 7: state_space = calculateStateSpace_7(components); break;
        default:
            assert(("Invalid switch combination", 0));
    }
    Model_diode_Topology& topology = state_space_cache.emplace_back(Model_diode_Topology{
        .components = components,
        .switches = switches,
        .state_space = std::move(state_space)});

    return *topology.state_space;
}

std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_0(Model_diode::Components const& c) //
{
    double L1 = c.L1;
    double L2 = c.L2;
    double R1 = c.R1;
    double R2 = c.R2;
    double R3 = c.R3;
    double R4 = c.R4;
    double R_D2 = c.R_D2;
    double R_D3 = c.R_D3;

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> K1{
        {0, L2},
        {1, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> K2{
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {L1, 0},
        {0, 0},
        {0, 0},
        {L1, 0},
        {L1, 0}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> A1{
        {0, -R4},
        {0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_INPUTS> B1{
        {-1, 0, 0, 0},
        {0, 0, 0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> C1{
        {0, 0},
        {0, 1},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_INPUTS> D1{
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, -1},
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 0, -1},
        {0, 0, 0, 0}};

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_1(Model_diode::Components const& c) //  S1
{
    double L1 = c.L1;
    double L2 = c.L2;
    double R1 = c.R1;
    double R2 = c.R2;
    double R3 = c.R3;
    double R4 = c.R4;
    double R_D2 = c.R_D2;
    double R_D3 = c.R_D3;

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> K1{
        {L1, 0},
        {0, L2}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> K2{
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {L1, 0}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> A1{
        {-R1 - R2 - R3, 0},
        {0, -R4}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_INPUTS> B1{
        {1, 0, 0, 1},
        {-1, 0, 0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> C1{
        {1, 0},
        {0, 1},
        {0, 0},
        {0, 0},
        {-R1, 0},
        {R2, 0},
        {0, 0},
        {0, 0},
        {0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_INPUTS> D1{
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 0, 0}};

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_2(Model_diode::Components const& c) //  S_D2
{
    double L1 = c.L1;
    double L2 = c.L2;
    double R1 = c.R1;
    double R2 = c.R2;
    double R3 = c.R3;
    double R4 = c.R4;
    double R_D2 = c.R_D2;
    double R_D3 = c.R_D3;

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> K1{
        {L1, 0},
        {0, L2}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> K2{
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {L1, 0},
        {L1, 0}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> A1{
        {-R3 - R_D2, 0},
        {0, -R4}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_INPUTS> B1{
        {0, -1, 0, 1},
        {-1, 0, 0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> C1{
        {1, 0},
        {0, 1},
        {1, 0},
        {0, 0},
        {-R_D2, 0},
        {0, 0},
        {0, 0},
        {R3, 0},
        {0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_INPUTS> D1{
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, -1, 0, 0},
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 0, -1},
        {0, 0, 0, 0}};

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_3(Model_diode::Components const& c) //  S1 S_D2
{
    double L1 = c.L1;
    double L2 = c.L2;
    double R1 = c.R1;
    double R2 = c.R2;
    double R3 = c.R3;
    double R4 = c.R4;
    double R_D2 = c.R_D2;
    double R_D3 = c.R_D3;

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> K1{
        {L1, 0},
        {0, L2}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> K2{
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {L1, 0}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> A1{
        {-(R3 * (R1 + R2 + R_D2) + R_D2 * (R1 + R2)) / (R1 + R2 + R_D2), 0},
        {0, -R4}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_INPUTS> B1{
        {R_D2 / (R1 + R2 + R_D2), -(R1 + R2) / (R1 + R2 + R_D2), 0, -(-R1 - R2 - R_D2) / (R1 + R2 + R_D2)},
        {-1, 0, 0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> C1{
        {1, 0},
        {0, 1},
        {R1 / (R1 + R2 + R_D2) + R2 / (R1 + R2 + R_D2), 0},
        {0, 0},
        {-R1 * R_D2 / (R1 + R2 + R_D2), 0},
        {R2 * R_D2 / (R1 + R2 + R_D2), 0},
        {0, 0},
        {0, 0},
        {0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_INPUTS> D1{
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {-1 / (R1 + R2 + R_D2), -1 / (R1 + R2 + R_D2), 0, 0},
        {0, 0, 0, 0},
        {-R1 / (R1 + R2 + R_D2) + 1, -R1 / (R1 + R2 + R_D2), 0, 0},
        {R2 / (R1 + R2 + R_D2), R2 / (R1 + R2 + R_D2), 0, 0},
        {1, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 0, 0}};

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_4(Model_diode::Components const& c) //  S_D3
{
    double L1 = c.L1;
    double L2 = c.L2;
    double R1 = c.R1;
    double R2 = c.R2;
    double R3 = c.R3;
    double R4 = c.R4;
    double R_D2 = c.R_D2;
    double R_D3 = c.R_D3;

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> K1{
        {L1, 0},
        {0, L2}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> K2{
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {L1, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {L1, 0}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> A1{
        {-R1 - R2 - R3 - R_D3, 0},
        {0, -R4}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_INPUTS> B1{
        {1, 0, 1, 1},
        {-1, 0, 0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> C1{
        {1, 0},
        {0, 1},
        {0, 0},
        {-1, 0},
        {R2 + R3, 0},
        {R2, 0},
        {0, 0},
        {-R_D3, 0},
        {0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_INPUTS> D1{
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, -1},
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {1, 0, 1, 0},
        {0, 0, 0, 0}};

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_5(Model_diode::Components const& c) //  S1 S_D3
{
    double L1 = c.L1;
    double L2 = c.L2;
    double R1 = c.R1;
    double R2 = c.R2;
    double R3 = c.R3;
    double R4 = c.R4;
    double R_D2 = c.R_D2;
    double R_D3 = c.R_D3;

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> K1{
        {L1, 0},
        {0, L2}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> K2{
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {L1, 0}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> A1{
        {-R1 - R2 - R3, 0},
        {0, -R4}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_INPUTS> B1{
        {1, 0, 0, 1},
        {-1, 0, 0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> C1{
        {1, 0},
        {0, 1},
        {0, 0},
        {0, 0},
        {-R1, 0},
        {R2, 0},
        {0, 0},
        {0, 0},
        {0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_INPUTS> D1{
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, -1 / R_D3, 0},
        {1, 0, 0, 0},
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 0, 0}};

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_6(Model_diode::Components const& c) //  S_D2 S_D3
{
    double L1 = c.L1;
    double L2 = c.L2;
    double R1 = c.R1;
    double R2 = c.R2;
    double R3 = c.R3;
    double R4 = c.R4;
    double R_D2 = c.R_D2;
    double R_D3 = c.R_D3;

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> K1{
        {L1, 0},
        {0, L2}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> K2{
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {L1, 0}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> A1{
        {-(R3 * (R1 + R2 + R_D2 + R_D3) + R_D2 * (R1 + R2 + R_D3)) / (R1 + R2 + R_D2 + R_D3), 0},
        {0, -R4}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_INPUTS> B1{
        {R_D2 / (R1 + R2 + R_D2 + R_D3), -(R1 + R2 + R_D3) / (R1 + R2 + R_D2 + R_D3), R_D2 / (R1 + R2 + R_D2 + R_D3), -(-R1 - R2 - R_D2 - R_D3) / (R1 + R2 + R_D2 + R_D3)},
        {-1, 0, 0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> C1{
        {1, 0},
        {0, 1},
        {R1 / (R1 + R2 + R_D2 + R_D3) + R2 / (R1 + R2 + R_D2 + R_D3) + R_D3 / (R1 + R2 + R_D2 + R_D3), 0},
        {-R_D2 / (R1 + R2 + R_D2 + R_D3), 0},
        {R2 * R_D2 / (R1 + R2 + R_D2 + R_D3) - R_D2 * (R1 / (R1 + R2 + R_D2 + R_D3) + R2 / (R1 + R2 + R_D2 + R_D3) + R_D3 / (R1 + R2 + R_D2 + R_D3)), 0},
        {R2 * R_D2 / (R1 + R2 + R_D2 + R_D3), 0},
        {0, 0},
        {-R_D2 * R_D3 / (R1 + R2 + R_D2 + R_D3), 0},
        {0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_INPUTS> D1{
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {-1 / (R1 + R2 + R_D2 + R_D3), -1 / (R1 + R2 + R_D2 + R_D3), -1 / (R1 + R2 + R_D2 + R_D3), 0},
        {-1 / (R1 + R2 + R_D2 + R_D3), -1 / (R1 + R2 + R_D2 + R_D3), -1 / (R1 + R2 + R_D2 + R_D3), 0},
        {R2 / (R1 + R2 + R_D2 + R_D3) + R_D2 / (R1 + R2 + R_D2 + R_D3), R2 / (R1 + R2 + R_D2 + R_D3) + R_D2 / (R1 + R2 + R_D2 + R_D3) - 1, R2 / (R1 + R2 + R_D2 + R_D3) + R_D2 / (R1 + R2 + R_D2 + R_D3), 0},
        {R2 / (R1 + R2 + R_D2 + R_D3), R2 / (R1 + R2 + R_D2 + R_D3), R2 / (R1 + R2 + R_D2 + R_D3), 0},
        {1, 0, 0, 0},
        {-R_D3 / (R1 + R2 + R_D2 + R_D3) + 1, -R_D3 / (R1 + R2 + R_D2 + R_D3), -R_D3 / (R1 + R2 + R_D2 + R_D3) + 1, 0},
        {0, 0, 0, 0}};

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

std::unique_ptr<Model_diode::StateSpaceMatrices> calculateStateSpace_7(Model_diode::Components const& c) //  S1 S_D2 S_D3
{
    double L1 = c.L1;
    double L2 = c.L2;
    double R1 = c.R1;
    double R2 = c.R2;
    double R3 = c.R3;
    double R4 = c.R4;
    double R_D2 = c.R_D2;
    double R_D3 = c.R_D3;

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> K1{
        {L1, 0},
        {0, L2}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> K2{
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
        {L1, 0}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_STATES> A1{
        {-(R3 * (R1 + R2 + R_D2) + R_D2 * (R1 + R2)) / (R1 + R2 + R_D2), 0},
        {0, -R4}};

    Eigen::Matrix<double, Model_diode::NUM_STATES, Model_diode::NUM_INPUTS> B1{
        {R_D2 / (R1 + R2 + R_D2), -(R1 + R2) / (R1 + R2 + R_D2), 0, -(-R1 - R2 - R_D2) / (R1 + R2 + R_D2)},
        {-1, 0, 0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_STATES> C1{
        {1, 0},
        {0, 1},
        {R1 / (R1 + R2 + R_D2) + R2 / (R1 + R2 + R_D2), 0},
        {0, 0},
        {-R1 * R_D2 / (R1 + R2 + R_D2), 0},
        {R2 * R_D2 / (R1 + R2 + R_D2), 0},
        {0, 0},
        {0, 0},
        {0, 0}};

    Eigen::Matrix<double, Model_diode::NUM_OUTPUTS, Model_diode::NUM_INPUTS> D1{
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {-1 / (R1 + R2 + R_D2), -1 / (R1 + R2 + R_D2), 0, 0},
        {0, 0, -1 / R_D3, 0},
        {-R1 / (R1 + R2 + R_D2) + 1, -R1 / (R1 + R2 + R_D2), 0, 0},
        {R2 / (R1 + R2 + R_D2), R2 / (R1 + R2 + R_D2), 0, 0},
        {1, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 0, 0}};

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}
