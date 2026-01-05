
#include "diode_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <fstream>
#include <format>
#include <memory>
#include "diode_matrices_json.h"

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

    // Diode D2
    double V_D2 = circuit.outputs.N_D2_pos - circuit.outputs.N_D2_neg;
    if (V_D2 > circuit.inputs.V_D2 && !circuit.switches.S_D2) {
        double V_D2_prev = prev_outputs.N_D2_pos - prev_outputs.N_D2_neg;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D2_prev, V_D2),
            .event_callback = [&]() {
                circuit.switches.S_D2.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D2 < 0 && circuit.switches.S_D2.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D2, circuit.outputs.I_R_D2),
            .event_callback = [&]() {
                circuit.switches.S_D2.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D3
    double V_D3 = circuit.outputs.N_D3_pos - circuit.outputs.N_D3_neg;
    if (V_D3 > circuit.inputs.V_D3 && !circuit.switches.S_D3) {
        double V_D3_prev = prev_outputs.N_D3_pos - prev_outputs.N_D3_neg;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D3_prev, V_D3),
            .event_callback = [&]() {
                circuit.switches.S_D3.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D3 < 0 && circuit.switches.S_D3.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D3, circuit.outputs.I_R_D3),
            .event_callback = [&]() {
                circuit.switches.S_D3.forceOutput(std::nullopt);
            }
        });
    }

    if (events.size() > 0) {
        return events.top();
    }
    return std::nullopt;
}

Model_diode::Model_diode(Components const& c)
    : components(c),
      _M_components_DO_NOT_TOUCH(c) {
}

void Model_diode::step(double dt, Inputs const& inputs_) {
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

void Model_diode::stepWithZeroCrossingDetection(double dt) {
    // No need to do anything
    if (dt < rlc2ss::MINIMUM_TIMESTEP) {
        return;
    }

    // Copy previous state and outputs if step needs to be redone
    Model_diode::States prev_state;
    Model_diode::Outputs prev_outputs;
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

void Model_diode::stepModel(double dt) {
    dt = std::max(dt, m_dt_resolution);
    // Update state-space matrices if needed
    if (components != _M_components_DO_NOT_TOUCH || switches.all() != _M_switches_DO_NOT_TOUCH.all() || !m_solver.initialized()) {
        assert(components.L1 != -1);
        assert(components.L2 != -1);
        assert(components.R1 != -1);
        assert(components.R2 != -1);
        assert(components.R3 != -1);
        assert(components.R4 != -1);
        assert(components.R_D2 != -1);
        assert(components.R_D3 != -1);
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
    states.I_L1 = outputs.I_L1;
    states.I_L2 = outputs.I_L2;
}

struct Model_diode_Topology {
    Model_diode::Components components;
    Model_diode::Switches switches;
    std::unique_ptr<Model_diode::StateSpaceMatrices> state_space;
};

void Model_diode::updateStateSpaceMatrices() {
    static std::vector<Model_diode_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_diode_Topology const& t) {
            return t.components == components && t.switches.all() == switches.all();
        });
    if (it != state_space_cache.end()) {
        m_ss = *it->state_space;
        return;
    }

    if (m_circuit_json.empty()) {
        m_circuit_json = nlohmann::json::parse(std::string(diode_matrices_json_hexdump, diode_matrices_json_hexdump + diode_matrices_json_hexdump_len));
    }
    assert(m_circuit_json.contains(std::to_string(switches.all())));

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = m_circuit_json[std::to_string(switches.all())].dump();
    s = rlc2ss::replace(s, "L1", std::format("({})", components.L1));
    s = rlc2ss::replace(s, "L2", std::format("({})", components.L2));
    s = rlc2ss::replace(s, "R1", std::format("({})", components.R1));
    s = rlc2ss::replace(s, "R2", std::format("({})", components.R2));
    s = rlc2ss::replace(s, "R3", std::format("({})", components.R3));
    s = rlc2ss::replace(s, "R4", std::format("({})", components.R4));
    s = rlc2ss::replace(s, "R_D2", std::format("({})", components.R_D2));
    s = rlc2ss::replace(s, "R_D3", std::format("({})", components.R_D3));

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

    Model_diode_Topology& topology = state_space_cache.emplace_back(Model_diode_Topology{
        .components = components,
        .switches = switches,
        .state_space = calcStateSpace(K1, A1, B1, K2, C1, D1)});

    m_ss = *topology.state_space;
}

uint64_t Model_diode::Switches::all() const {
    return 0 |
        (S1 << 0) |
        (S_D2 << 1) |
        (S_D3 << 2);
}

double Model_diode::Switches::smallestDelay() {
    return std::min({double(rlc2ss::OnOffDelay::MAX_DELAY),
                    S1.pendingTime(),
                    S_D2.pendingTime(),
                    S_D3.pendingTime()});
}

void Model_diode::Switches::step(double dt) {
    S1.step(dt);
    S_D2.step(dt);
    S_D3.step(dt);
}
