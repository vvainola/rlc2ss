
#include "converter_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <fstream>
#include <format>
#include <memory>
#include "converter_matrices_json.h"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

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

    // Diode D_a_n
    double V_D_a_n = circuit.outputs.N_dc_n - circuit.outputs.N_c_a;
    if (V_D_a_n > circuit.inputs.V_D_a_n && !circuit.switches.S_D_a_n) {
        double V_D_a_n_prev = prev_outputs.N_dc_n - prev_outputs.N_c_a;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_a_n_prev, V_D_a_n),
            .event_callback = [&]() {
                circuit.switches.S_D_a_n.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_a_n < 0 && circuit.switches.S_D_a_n.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_a_n, circuit.outputs.I_R_D_a_n),
            .event_callback = [&]() {
                circuit.switches.S_D_a_n.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_a_p
    double V_D_a_p = circuit.outputs.N_c_a - circuit.outputs.N_dc_p;
    if (V_D_a_p > circuit.inputs.V_D_a_p && !circuit.switches.S_D_a_p) {
        double V_D_a_p_prev = prev_outputs.N_c_a - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_a_p_prev, V_D_a_p),
            .event_callback = [&]() {
                circuit.switches.S_D_a_p.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_a_p < 0 && circuit.switches.S_D_a_p.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_a_p, circuit.outputs.I_R_D_a_p),
            .event_callback = [&]() {
                circuit.switches.S_D_a_p.forceOutput(std::nullopt);
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
                circuit.switches.S_D_b_n.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_b_n < 0 && circuit.switches.S_D_b_n.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_b_n, circuit.outputs.I_R_D_b_n),
            .event_callback = [&]() {
                circuit.switches.S_D_b_n.forceOutput(std::nullopt);
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
                circuit.switches.S_D_b_p.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_b_p < 0 && circuit.switches.S_D_b_p.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_b_p, circuit.outputs.I_R_D_b_p),
            .event_callback = [&]() {
                circuit.switches.S_D_b_p.forceOutput(std::nullopt);
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
                circuit.switches.S_D_c_n.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_c_n < 0 && circuit.switches.S_D_c_n.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_c_n, circuit.outputs.I_R_D_c_n),
            .event_callback = [&]() {
                circuit.switches.S_D_c_n.forceOutput(std::nullopt);
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
                circuit.switches.S_D_c_p.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_c_p < 0 && circuit.switches.S_D_c_p.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_c_p, circuit.outputs.I_R_D_c_p),
            .event_callback = [&]() {
                circuit.switches.S_D_c_p.forceOutput(std::nullopt);
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

struct Model_converter_Topology {
    Model_converter::Components components;
    Model_converter::Switches switches;
    std::unique_ptr<Model_converter::StateSpaceMatrices> state_space;
};

void Model_converter::updateStateSpaceMatrices() {
    static std::vector<Model_converter_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_converter_Topology const& t) {
            return t.components == components && t.switches.all() == switches.all();
        });
    if (it != state_space_cache.end()) {
        m_ss = *it->state_space;
        return;
    }
    std::string netlist = "R_p_p 0 N_dc_p 1E3 \nR_n_p N_dc_n 0 1E3 \nV_dc _net0 N_dc_n DC 1 \nD_a_p N_c_a N_dc_p \nD_b_p N_c_b N_dc_p \nS_a_p N_c_a N_dc_p _net1 _net2 \nS_c_p N_c_c N_dc_p _net3 _net4 \nD_c_p N_c_c N_dc_p \nS_b_p N_c_b N_dc_p _net5 _net6 \nS_a_n N_dc_n N_c_a _net7 _net8 \nD_a_n N_dc_n N_c_a \nS_b_n N_dc_n N_c_b _net9 _net10 \nD_b_n N_dc_n N_c_b \nS_c_n N_dc_n N_c_c _net11 _net12 \nD_c_n N_dc_n N_c_c \nV_a _net13 _net14 DC 0 SIN(0 1 1K 0 0 0) AC 1 ACPHASE 0 \nV_c _net15 _net14 DC 0 SIN(0 1 1K 0 0 0) AC 1 ACPHASE 0 \nV_b _net16 _net14 DC 0 SIN(0 1 1K 0 0 0) AC 1 ACPHASE 0 \nL_b _net17 _net16 1M \nL_a _net18 _net13 1M \nL_c _net19 _net15 1M \nC_p 0 _net20 10E-3;I; \nC_n _net21 0 10E-3;I; \nR_n_s N_dc_n _net21 10E-3 \nR_p_s _net20 N_dc_p 10E-3 \nR_a N_c_a _net18 10E-3 \nR_b N_c_b _net17 10E-3 \nR_c N_c_c _net19 10E-3 \nR_dc _net0 N_dc_p 1;I; ";
    std::unordered_map<std::string, double> component_values;
	component_values["C_n"] = components.C_n;
	component_values["C_p"] = components.C_p;
	component_values["L_a"] = components.L_a;
	component_values["L_b"] = components.L_b;
	component_values["L_c"] = components.L_c;
	component_values["R_D_a_n"] = components.R_D_a_n;
	component_values["R_D_a_p"] = components.R_D_a_p;
	component_values["R_D_b_n"] = components.R_D_b_n;
	component_values["R_D_b_p"] = components.R_D_b_p;
	component_values["R_D_c_n"] = components.R_D_c_n;
	component_values["R_D_c_p"] = components.R_D_c_p;
	component_values["R_a"] = components.R_a;
	component_values["R_b"] = components.R_b;
	component_values["R_c"] = components.R_c;
	component_values["R_dc"] = components.R_dc;
	component_values["R_n_p"] = components.R_n_p;
	component_values["R_n_s"] = components.R_n_s;
	component_values["R_p_p"] = components.R_p_p;
	component_values["R_p_s"] = components.R_p_s;
    rlc2ss::StateSpaceMatrices ss = rlc2ss::formStateSpaceMatrices(netlist, int(switches.all()), component_values);

    // Create eigen matrices
    Eigen::Matrix<double, NUM_STATES, NUM_STATES, Eigen::RowMajor> K1(rlc2ss::getCommaDelimitedValues(ss.K1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES, Eigen::RowMajor> K2(rlc2ss::getCommaDelimitedValues(ss.K2).data());
    Eigen::Matrix<double, NUM_STATES, NUM_STATES, Eigen::RowMajor> A1(rlc2ss::getCommaDelimitedValues(ss.A1).data());
    Eigen::Matrix<double, NUM_STATES, NUM_INPUTS, Eigen::RowMajor> B1(rlc2ss::getCommaDelimitedValues(ss.B1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES, Eigen::RowMajor> C1(rlc2ss::getCommaDelimitedValues(ss.C1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_INPUTS, Eigen::RowMajor> D1(rlc2ss::getCommaDelimitedValues(ss.D1).data());

    Model_converter_Topology& topology = state_space_cache.emplace_back(Model_converter_Topology{
        .components = components,
        .switches = switches,
        .state_space = calcStateSpace(K1, A1, B1, K2, C1, D1)});

    m_ss = *topology.state_space;
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
