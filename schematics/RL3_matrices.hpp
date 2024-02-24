
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
#include <assert.h>

class Model_RL3 {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    union Switches;
    struct StateSpaceMatrices;
    static StateSpaceMatrices calculateStateSpace(Components const& components, Switches switches);

    Model_RL3() {}
    Model_RL3(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 3;
    static inline constexpr size_t NUM_OUTPUTS = 3;
    static inline constexpr size_t NUM_STATES = 3;
    static inline constexpr size_t NUM_SWITCHES = 0;

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
        states.I_L_a = outputs.I_L_a;
        states.I_L_b = outputs.I_L_b;
        states.I_L_c = outputs.I_L_c;
    }

    struct Components {
        double L_a = -1;
        double L_b = -1;
        double L_c = -1;
        double R_a = -1;
        double R_b = -1;
        double R_c = -1;
        double Kab = -1;
        double Kbc = -1;
        double Kca = -1;

        bool operator==(Components const& other) const {
            return
                L_a == other.L_a &&
                L_b == other.L_b &&
                L_c == other.L_c &&
                R_a == other.R_a &&
                R_b == other.R_b &&
                R_c == other.R_c &&
                Kab == other.Kab &&
                Kbc == other.Kbc &&
                Kca == other.Kca;
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
            double I_L_a;
            double I_L_b;
            double I_L_c;
        };
        Eigen::Vector<double, NUM_STATES> data;
    };

    union Inputs {
        Inputs() {
            data.setZero();
        }
        struct {
            double V_a;
            double V_b;
            double V_c;
        };
        Eigen::Vector<double, NUM_INPUTS> data;
    };

    union Outputs {
        Outputs() {
            data.setZero();
        }
        struct {
            double I_L_a;
            double I_L_b;
            double I_L_c;
        };
        Eigen::Vector<double, NUM_OUTPUTS> data;
    };

    union Switches {
        struct {

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
    Inputs m_inputs;
    Integrator<Eigen::Vector<double, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components m_components_DO_NOT_TOUCH;
    Switches m_switches_DO_NOT_TOUCH = {.all = 0};
    Eigen::Vector<double, NUM_STATES> m_Bu; // Bu term in "dxdt = Ax + Bu"
    double m_dt_prev = 0;
    double m_dt_implicit_limit = 0;

    static_assert(sizeof(double) * NUM_STATES == sizeof(States));
    static_assert(sizeof(double) * NUM_INPUTS == sizeof(Inputs));
    static_assert(sizeof(double) * NUM_OUTPUTS == sizeof(Outputs));
};

#pragma warning(default : 4127) // conditional expression is constant
#pragma warning(default : 4189) // local variable is initialized but not referenced
#pragma warning(default : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(default : 4408) // anonymous struct did not declare any data members
#pragma warning(default : 5054) // operator '&': deprecated between enumerations of different types
