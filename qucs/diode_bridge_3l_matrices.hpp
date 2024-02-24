
#pragma once

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include "integrator.h"
#include "nlohmann/json.hpp"
#include <assert.h>

class Model_diode_bridge_3l {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    union Switches;
    struct StateSpaceMatrices;

    Model_diode_bridge_3l() {}
    Model_diode_bridge_3l(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 4;
    static inline constexpr size_t NUM_OUTPUTS = 25;
    static inline constexpr size_t NUM_STATES = 19;
    static inline constexpr size_t NUM_SWITCHES = 9;

    Eigen::Vector<double, NUM_STATES> dxdt(Eigen::Vector<double, NUM_STATES> const& state, double /*t*/) const {
        return m_ss.A * state + m_Bu;
    }

    // Implicit Tustin integration is used for timesteps larger than this.
    void setImplicitIntegrationLimit(double dt) {
        m_solver.enableInverseMatrixCaching(true);
        m_dt_implicit_limit = dt;
    }

    void step(double dt, Inputs const& inputs_) {
        inputs.data = inputs_.data;
        // Update state-space matrices if needed
        if (components != m_components_DO_NOT_TOUCH || switches.all != m_switches_DO_NOT_TOUCH.all) {
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
            assert(components.C_dc_n1 != -1);
            assert(components.C_dc_n2 != -1);
            assert(components.C_dc_p1 != -1);
            assert(components.C_dc_p2 != -1);
            assert(components.C_f_a != -1);
            assert(components.C_f_b != -1);
            assert(components.C_f_c != -1);
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
            assert(components.R_dc_src_s != -1);
            assert(components.R_dc_src_p != -1);
            assert(components.R_f_a != -1);
            assert(components.R_f_b != -1);
            assert(components.R_f_c != -1);
            assert(components.R_grid_a != -1);
            assert(components.R_grid_b != -1);
            assert(components.R_grid_c != -1);
            assert(components.R_src_a != -1);
            assert(components.R_src_b != -1);
            assert(components.R_src_c != -1);
            m_components_DO_NOT_TOUCH = components;
            m_switches_DO_NOT_TOUCH.all = switches.all;
            m_ss = calculateStateSpace(components, switches);
            m_solver.updateJacobian(m_ss.A);
            // Solve one step with backward euler to reduce numerical oscillations
            m_Bu = m_ss.B * inputs.data;
            states.data = m_solver.stepBackwardEuler(*this, states.data, 0.0, dt);
        } else {
            m_Bu = m_ss.B * inputs.data;

            if (m_dt_implicit_limit > 0) {
                // Solve with tustin as multiples of implicit limit and the remaining time with runge-kutta so
                // that the matrix inverses required for implicit integration can be cached for common timesteps
                // and weird small remainders are solved with adaptive integration.
                double multiple = dt / m_dt_implicit_limit;
                if (std::abs(std::round(multiple) - multiple) > 1e-6) {
                    double dt1 = std::floor(multiple) * m_dt_implicit_limit;
                    double dt2 = (multiple - std::floor(multiple)) * m_dt_implicit_limit;
                    states.data = m_solver.stepTustin(*this, states.data, 0.0, dt1);
                    states.data = m_solver.stepRungeKuttaFehlberg(*this, states.data, 0.0, dt2);
                } else {
                    states.data = m_solver.stepTustin(*this, states.data, 0.0, std::round(multiple) * m_dt_implicit_limit);
                }
            } else {
                states.data = m_solver.stepTustin(*this, states.data, 0.0, dt);
            }
        }
        m_dt_prev = dt;

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

    struct Components {
        double L_conv_a = -1;
        double L_conv_b = -1;
        double L_conv_c = -1;
        double L_dc_n = -1;
        double L_dc_p = -1;
        double L_dc_src = -1;
        double L_grid_a = -1;
        double L_grid_b = -1;
        double L_grid_c = -1;
        double L_src_a = -1;
        double L_src_b = -1;
        double L_src_c = -1;
        double C_dc_n1 = -1;
        double C_dc_n2 = -1;
        double C_dc_p1 = -1;
        double C_dc_p2 = -1;
        double C_f_a = -1;
        double C_f_b = -1;
        double C_f_c = -1;
        double R_conv_a = -1;
        double R_conv_b = -1;
        double R_conv_c = -1;
        double R_dc_pn1 = -1;
        double R_dc_pn2 = -1;
        double R_dc_pp1 = -1;
        double R_dc_pp2 = -1;
        double R_dc_sn1 = -1;
        double R_dc_sn2 = -1;
        double R_dc_sp1 = -1;
        double R_dc_sp2 = -1;
        double R_dc_src_s = -1;
        double R_dc_src_p = -1;
        double R_f_a = -1;
        double R_f_b = -1;
        double R_f_c = -1;
        double R_grid_a = -1;
        double R_grid_b = -1;
        double R_grid_c = -1;
        double R_src_a = -1;
        double R_src_b = -1;
        double R_src_c = -1;

        bool operator==(Components const& other) const {
            return
                L_conv_a == other.L_conv_a &&
                L_conv_b == other.L_conv_b &&
                L_conv_c == other.L_conv_c &&
                L_dc_n == other.L_dc_n &&
                L_dc_p == other.L_dc_p &&
                L_dc_src == other.L_dc_src &&
                L_grid_a == other.L_grid_a &&
                L_grid_b == other.L_grid_b &&
                L_grid_c == other.L_grid_c &&
                L_src_a == other.L_src_a &&
                L_src_b == other.L_src_b &&
                L_src_c == other.L_src_c &&
                C_dc_n1 == other.C_dc_n1 &&
                C_dc_n2 == other.C_dc_n2 &&
                C_dc_p1 == other.C_dc_p1 &&
                C_dc_p2 == other.C_dc_p2 &&
                C_f_a == other.C_f_a &&
                C_f_b == other.C_f_b &&
                C_f_c == other.C_f_c &&
                R_conv_a == other.R_conv_a &&
                R_conv_b == other.R_conv_b &&
                R_conv_c == other.R_conv_c &&
                R_dc_pn1 == other.R_dc_pn1 &&
                R_dc_pn2 == other.R_dc_pn2 &&
                R_dc_pp1 == other.R_dc_pp1 &&
                R_dc_pp2 == other.R_dc_pp2 &&
                R_dc_sn1 == other.R_dc_sn1 &&
                R_dc_sn2 == other.R_dc_sn2 &&
                R_dc_sp1 == other.R_dc_sp1 &&
                R_dc_sp2 == other.R_dc_sp2 &&
                R_dc_src_s == other.R_dc_src_s &&
                R_dc_src_p == other.R_dc_src_p &&
                R_f_a == other.R_f_a &&
                R_f_b == other.R_f_b &&
                R_f_c == other.R_f_c &&
                R_grid_a == other.R_grid_a &&
                R_grid_b == other.R_grid_b &&
                R_grid_c == other.R_grid_c &&
                R_src_a == other.R_src_a &&
                R_src_b == other.R_src_b &&
                R_src_c == other.R_src_c;
        }

        bool operator!=(Components const& other) const {
            return !(*this == other);
        }
    };

    union States {
        States() {
            data.setZero();
        }
        struct {
            double I_L_conv_a;
            double I_L_conv_b;
            double I_L_conv_c;
            double I_L_dc_n;
            double I_L_dc_p;
            double I_L_dc_src;
            double I_L_grid_a;
            double I_L_grid_b;
            double I_L_grid_c;
            double I_L_src_a;
            double I_L_src_b;
            double I_L_src_c;
            double V_C_dc_n1;
            double V_C_dc_n2;
            double V_C_dc_p1;
            double V_C_dc_p2;
            double V_C_f_a;
            double V_C_f_b;
            double V_C_f_c;
        };
        Eigen::Vector<double, NUM_STATES> data;
    };

    union Inputs {
        Inputs() {
            data.setZero();
        }
        struct {
            double V_dc_src;
            double V_src_a;
            double V_src_b;
            double V_src_c;
        };
        Eigen::Vector<double, NUM_INPUTS> data;
    };

    union Outputs {
        Outputs() {
            data.setZero();
        }
        struct {
            double I_L_conv_a;
            double I_L_conv_b;
            double I_L_conv_c;
            double I_L_dc_n;
            double I_L_dc_p;
            double I_L_dc_src;
            double I_L_grid_a;
            double I_L_grid_b;
            double I_L_grid_c;
            double I_L_src_a;
            double I_L_src_b;
            double I_L_src_c;
            double N_conv_a;
            double N_conv_b;
            double N_conv_c;
            double N_dc_0;
            double N_dc_n;
            double N_dc_p;
            double V_C_dc_n1;
            double V_C_dc_n2;
            double V_C_dc_p1;
            double V_C_dc_p2;
            double V_C_f_a;
            double V_C_f_b;
            double V_C_f_c;
        };
        Eigen::Vector<double, NUM_OUTPUTS> data;
    };

    union Switches {
        struct {
            uint32_t S_0_a : 1;
            uint32_t S_0_b : 1;
            uint32_t S_0_c : 1;
            uint32_t S_n_a : 1;
            uint32_t S_n_b : 1;
            uint32_t S_n_c : 1;
            uint32_t S_p_a : 1;
            uint32_t S_p_b : 1;
            uint32_t S_p_c : 1;
        };
        uint32_t all;
    };

    struct StateSpaceMatrices {
        Eigen::Matrix<double, NUM_STATES, NUM_STATES> A;
        Eigen::Matrix<double, NUM_STATES, NUM_INPUTS> B;
        Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES> C;
        Eigen::Matrix<double, NUM_OUTPUTS, NUM_INPUTS> D;
    };

    Components components;
    Inputs inputs;
    States states;
    Outputs outputs;
    Switches switches = {.all = 0};

  private:
    StateSpaceMatrices calculateStateSpace(Components const& components, Switches switches);

    Integrator<Eigen::Vector<double, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components m_components_DO_NOT_TOUCH;
    Switches m_switches_DO_NOT_TOUCH = {.all = 0};
    Eigen::Vector<double, NUM_STATES> m_Bu; // Bu term in "dxdt = Ax + Bu"
    double m_dt_prev = 0;
    double m_dt_implicit_limit = 0;
    // The json file with symbolic intermediate matrices
    nlohmann::json m_circuit_json;

    static_assert(sizeof(double) * NUM_STATES == sizeof(States));
    static_assert(sizeof(double) * NUM_INPUTS == sizeof(Inputs));
    static_assert(sizeof(double) * NUM_OUTPUTS == sizeof(Outputs));
};

#pragma warning(default : 4127) // conditional expression is constant
#pragma warning(default : 4189) // local variable is initialized but not referenced
#pragma warning(default : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(default : 4408) // anonymous struct did not declare any data members
#pragma warning(default : 5054) // operator '&': deprecated between enumerations of different types
