
#include "diode_bridge_3l_matrices.hpp"
#include "rlc2ss.h"
#include <optional>
#include <fstream>
#include <format>
#include <memory>
#include "diode_bridge_3l_matrices_json.h"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<Model_diode_bridge_3l::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_STATES, Model_diode_bridge_3l::NUM_STATES> const& K1,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_STATES, Model_diode_bridge_3l::NUM_STATES> const& A1,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_STATES, Model_diode_bridge_3l::NUM_INPUTS> const& B1,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_OUTPUTS, Model_diode_bridge_3l::NUM_STATES> const& K2,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_OUTPUTS, Model_diode_bridge_3l::NUM_STATES> const& C1,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_OUTPUTS, Model_diode_bridge_3l::NUM_INPUTS> const& D1) {
    auto ss = std::make_unique<Model_diode_bridge_3l::StateSpaceMatrices>();
    ss->A = K1.partialPivLu().solve(A1);
    ss->B = K1.partialPivLu().solve(B1);
    ss->C = (C1 + K2 * ss->A);
    ss->D = (D1 + K2 * ss->B);
    return ss;
}

static std::optional<rlc2ss::ZeroCrossingEvent> checkZeroCrossingEvents(Model_diode_bridge_3l& circuit, Model_diode_bridge_3l::Outputs const& prev_outputs) {
    std::priority_queue<rlc2ss::ZeroCrossingEvent,
                        std::vector<rlc2ss::ZeroCrossingEvent>,
                        std::greater<rlc2ss::ZeroCrossingEvent>>
        events;

    // Diode D_n_a
    double V_D_n_a = circuit.outputs.N_dc_n - circuit.outputs.N_conv_a;
    if (V_D_n_a > circuit.inputs.V_D_n_a && !circuit.switches.S_D_n_a) {
        double V_D_n_a_prev = prev_outputs.N_dc_n - prev_outputs.N_conv_a;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_n_a_prev, V_D_n_a),
            .event_callback = [&]() {
                circuit.switches.S_D_n_a.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_n_a < 0 && circuit.switches.S_D_n_a.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_n_a, circuit.outputs.I_R_D_n_a),
            .event_callback = [&]() {
                circuit.switches.S_D_n_a.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_n_b
    double V_D_n_b = circuit.outputs.N_dc_n - circuit.outputs.N_conv_b;
    if (V_D_n_b > circuit.inputs.V_D_n_b && !circuit.switches.S_D_n_b) {
        double V_D_n_b_prev = prev_outputs.N_dc_n - prev_outputs.N_conv_b;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_n_b_prev, V_D_n_b),
            .event_callback = [&]() {
                circuit.switches.S_D_n_b.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_n_b < 0 && circuit.switches.S_D_n_b.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_n_b, circuit.outputs.I_R_D_n_b),
            .event_callback = [&]() {
                circuit.switches.S_D_n_b.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_n_c
    double V_D_n_c = circuit.outputs.N_dc_n - circuit.outputs.N_conv_c;
    if (V_D_n_c > circuit.inputs.V_D_n_c && !circuit.switches.S_D_n_c) {
        double V_D_n_c_prev = prev_outputs.N_dc_n - prev_outputs.N_conv_c;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_n_c_prev, V_D_n_c),
            .event_callback = [&]() {
                circuit.switches.S_D_n_c.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_n_c < 0 && circuit.switches.S_D_n_c.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_n_c, circuit.outputs.I_R_D_n_c),
            .event_callback = [&]() {
                circuit.switches.S_D_n_c.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_p_a
    double V_D_p_a = circuit.outputs.N_conv_a - circuit.outputs.N_dc_p;
    if (V_D_p_a > circuit.inputs.V_D_p_a && !circuit.switches.S_D_p_a) {
        double V_D_p_a_prev = prev_outputs.N_conv_a - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_p_a_prev, V_D_p_a),
            .event_callback = [&]() {
                circuit.switches.S_D_p_a.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_p_a < 0 && circuit.switches.S_D_p_a.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_p_a, circuit.outputs.I_R_D_p_a),
            .event_callback = [&]() {
                circuit.switches.S_D_p_a.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_p_b
    double V_D_p_b = circuit.outputs.N_conv_b - circuit.outputs.N_dc_p;
    if (V_D_p_b > circuit.inputs.V_D_p_b && !circuit.switches.S_D_p_b) {
        double V_D_p_b_prev = prev_outputs.N_conv_b - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_p_b_prev, V_D_p_b),
            .event_callback = [&]() {
                circuit.switches.S_D_p_b.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_p_b < 0 && circuit.switches.S_D_p_b.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_p_b, circuit.outputs.I_R_D_p_b),
            .event_callback = [&]() {
                circuit.switches.S_D_p_b.forceOutput(std::nullopt);
            }
        });
    }

    // Diode D_p_c
    double V_D_p_c = circuit.outputs.N_conv_c - circuit.outputs.N_dc_p;
    if (V_D_p_c > circuit.inputs.V_D_p_c && !circuit.switches.S_D_p_c) {
        double V_D_p_c_prev = prev_outputs.N_conv_c - prev_outputs.N_dc_p;
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(V_D_p_c_prev, V_D_p_c),
            .event_callback = [&]() {
                circuit.switches.S_D_p_c.forceOutput(true);
            }
        });
    }
    if (circuit.outputs.I_R_D_p_c < 0 && circuit.switches.S_D_p_c.outputForced()) {
        events.push(rlc2ss::ZeroCrossingEvent{
            .time = rlc2ss::calcZeroCrossingTime(prev_outputs.I_R_D_p_c, circuit.outputs.I_R_D_p_c),
            .event_callback = [&]() {
                circuit.switches.S_D_p_c.forceOutput(std::nullopt);
            }
        });
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

struct Model_diode_bridge_3l_Topology {
    Model_diode_bridge_3l::Components components;
    Model_diode_bridge_3l::Switches switches;
    std::unique_ptr<Model_diode_bridge_3l::StateSpaceMatrices> state_space;
};

void Model_diode_bridge_3l::updateStateSpaceMatrices() {
    static std::vector<Model_diode_bridge_3l_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_diode_bridge_3l_Topology const& t) {
            return t.components == components && t.switches.all() == switches.all();
        });
    if (it != state_space_cache.end()) {
        m_ss = *it->state_space;
        return;
    }
    std::string netlist = "V_src_a _net0 0 DC 1 \nV_src_b _net1 0 DC 1 \nV_src_c _net2 0 DC 1 \nS_0_a N_conv_a N_dc_0 _net4 _net5 \nS_0_b N_conv_b N_dc_0 _net6 _net7 \nS_0_c N_conv_c N_dc_0 _net8 _net9 \nV_dc_src _net22 _net23 DC 1 \nR_dc_src_p _net25 _net24 1E3 \nL_dc_src _net22 _net24 10E-6 \nR_dc_pp2 _net25 _net27 1E3 \nR_dc_pn2 _net27 _net23 1E3 \nR_dc_sp2 _net25 _net26 1E-3 \nR_dc_sn2 _net27 _net28 1E-3 \nR_dc_pp1 N_dc_p N_dc_0 1E3 \nR_dc_pn1 N_dc_0 N_dc_n 1E3 \nR_dc_sp1 N_dc_p _net3 1E-3 \nR_dc_sn1 N_dc_0 _net29 1E-3 \nC_dc_p2 _net26 _net27 10E-3 \nC_dc_p1 _net3 N_dc_0 10E-3 \nC_dc_n1 _net29 N_dc_n 10E-3 \nC_dc_n2 _net28 _net23 10E-3 \nL_dc_n _net23 N_dc_n 1E-6 \nL_dc_p _net25 N_dc_p 1E-6 \nR_conv_a N_conv_a V2_a 1E-3 \nR_conv_b N_conv_b V2_b 1E-3 \nR_conv_c N_conv_c V2_c 1E-3 \nR_grid_a N_cap_a _net10 1E-3 \nR_grid_b N_cap_b _net12 1E-3 \nR_grid_c N_cap_c _net14 1E-3 \nR_src_a _net11 _net16 1E-3 \nR_src_b _net13 _net17 1E-3 \nR_src_c _net15 _net18 1E-3 \nL_conv_a V2_a N_cap_a 1E-6 \nL_conv_b V2_b N_cap_b 1E-6 \nL_conv_c V2_c N_cap_c 1E-6 \nL_grid_a _net10 _net11 1E-6 \nL_grid_b _net12 _net13 1E-6 \nL_grid_c _net14 _net15 1E-6 \nL_src_a _net16 _net0 1E-6 \nL_src_b _net17 _net1 1E-6 \nL_src_c _net18 _net2 1E-6 \nR_f_a _net19 N_cap_0 1E-3 \nR_f_b _net20 N_cap_0 1E-3 \nR_f_c _net21 N_cap_0 1E-3 \nC_f_a N_cap_a _net19 1E-3 \nC_f_b N_cap_b _net20 1E-3 \nC_f_c N_cap_c _net21 1E-3 \nD_p_a N_conv_a N_dc_p \nD_p_b N_conv_b N_dc_p \nD_p_c N_conv_c N_dc_p \nD_n_a N_dc_n N_conv_a \nD_n_b N_dc_n N_conv_b \nD_n_c N_dc_n N_conv_c \nR_dc_src_s _net25 _net24 1 ";
    std::unordered_map<std::string, double> component_values;
	component_values["C_dc_n1"] = components.C_dc_n1;
	component_values["C_dc_n2"] = components.C_dc_n2;
	component_values["C_dc_p1"] = components.C_dc_p1;
	component_values["C_dc_p2"] = components.C_dc_p2;
	component_values["C_f_a"] = components.C_f_a;
	component_values["C_f_b"] = components.C_f_b;
	component_values["C_f_c"] = components.C_f_c;
	component_values["L_conv_a"] = components.L_conv_a;
	component_values["L_conv_b"] = components.L_conv_b;
	component_values["L_conv_c"] = components.L_conv_c;
	component_values["L_dc_n"] = components.L_dc_n;
	component_values["L_dc_p"] = components.L_dc_p;
	component_values["L_dc_src"] = components.L_dc_src;
	component_values["L_grid_a"] = components.L_grid_a;
	component_values["L_grid_b"] = components.L_grid_b;
	component_values["L_grid_c"] = components.L_grid_c;
	component_values["L_src_a"] = components.L_src_a;
	component_values["L_src_b"] = components.L_src_b;
	component_values["L_src_c"] = components.L_src_c;
	component_values["R_D_n_a"] = components.R_D_n_a;
	component_values["R_D_n_b"] = components.R_D_n_b;
	component_values["R_D_n_c"] = components.R_D_n_c;
	component_values["R_D_p_a"] = components.R_D_p_a;
	component_values["R_D_p_b"] = components.R_D_p_b;
	component_values["R_D_p_c"] = components.R_D_p_c;
	component_values["R_conv_a"] = components.R_conv_a;
	component_values["R_conv_b"] = components.R_conv_b;
	component_values["R_conv_c"] = components.R_conv_c;
	component_values["R_dc_pn1"] = components.R_dc_pn1;
	component_values["R_dc_pn2"] = components.R_dc_pn2;
	component_values["R_dc_pp1"] = components.R_dc_pp1;
	component_values["R_dc_pp2"] = components.R_dc_pp2;
	component_values["R_dc_sn1"] = components.R_dc_sn1;
	component_values["R_dc_sn2"] = components.R_dc_sn2;
	component_values["R_dc_sp1"] = components.R_dc_sp1;
	component_values["R_dc_sp2"] = components.R_dc_sp2;
	component_values["R_dc_src_p"] = components.R_dc_src_p;
	component_values["R_dc_src_s"] = components.R_dc_src_s;
	component_values["R_f_a"] = components.R_f_a;
	component_values["R_f_b"] = components.R_f_b;
	component_values["R_f_c"] = components.R_f_c;
	component_values["R_grid_a"] = components.R_grid_a;
	component_values["R_grid_b"] = components.R_grid_b;
	component_values["R_grid_c"] = components.R_grid_c;
	component_values["R_src_a"] = components.R_src_a;
	component_values["R_src_b"] = components.R_src_b;
	component_values["R_src_c"] = components.R_src_c;
    rlc2ss::StateSpaceMatrices ss = rlc2ss::formStateSpaceMatrices(netlist, int(switches.all()), component_values);

    // Create eigen matrices
    Eigen::Matrix<double, NUM_STATES, NUM_STATES, Eigen::RowMajor> K1(rlc2ss::getCommaDelimitedValues(ss.K1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES, Eigen::RowMajor> K2(rlc2ss::getCommaDelimitedValues(ss.K2).data());
    Eigen::Matrix<double, NUM_STATES, NUM_STATES, Eigen::RowMajor> A1(rlc2ss::getCommaDelimitedValues(ss.A1).data());
    Eigen::Matrix<double, NUM_STATES, NUM_INPUTS, Eigen::RowMajor> B1(rlc2ss::getCommaDelimitedValues(ss.B1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES, Eigen::RowMajor> C1(rlc2ss::getCommaDelimitedValues(ss.C1).data());
    Eigen::Matrix<double, NUM_OUTPUTS, NUM_INPUTS, Eigen::RowMajor> D1(rlc2ss::getCommaDelimitedValues(ss.D1).data());

    Model_diode_bridge_3l_Topology& topology = state_space_cache.emplace_back(Model_diode_bridge_3l_Topology{
        .components = components,
        .switches = switches,
        .state_space = calcStateSpace(K1, A1, B1, K2, C1, D1)});

    m_ss = *topology.state_space;
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
        (S_D_p_c << 8);
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
                    S_D_p_c.pendingTime()});
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
}
