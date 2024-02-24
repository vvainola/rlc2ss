
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

class Model_diode_bridge {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    union Switches;
    struct StateSpaceMatrices;

    Model_diode_bridge() {}
    Model_diode_bridge(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 3;
    static inline constexpr size_t NUM_OUTPUTS = 9;
    static inline constexpr size_t NUM_STATES = 4;
    static inline constexpr size_t NUM_SWITCHES = 6;

    Eigen::Vector<double, NUM_STATES> dxdt(Eigen::Vector<double, NUM_STATES> const& state, double /*t*/) const {
        return m_ss.A * state + m_Bu;
    }

    enum class TimestepErrorCorrectionMode {
        // Ignore error in timestep length that is not a multiple of timestep resolution. Use this if
        // e.g. resolution is 0.1e-6 and the variation in timestep length is a multiple of that and
        // there should not ever be any error although floating point roundings may cause some.
        NONE,
        // Round the used timestep to closest multiple of resolution and store the error to accumulator
        // so that the timestep length error will be corrected when accumulator becomes a multiple of the
        // timestep resolution.
        ACCUMULATE,
        // The timestep length that is not a multiple of timestep resolution will be integrated with
        // adaptive step size runge-kutta-fehlberg. E.g. If resolution 1us and timestep is 12.1us,
        // 12 us will be solved with Tustin and remaining 0.1us with RKF to avoid calculating jacobian
        // inverse for very small timesteps
        INTEGRATE_ADAPTIVE
    };

    void setTimestepResolution(double dt, TimestepErrorCorrectionMode mode) {
        m_solver.enableInverseMatrixCaching(true);
        m_dt_resolution = dt;
        m_dt_correction_mode = mode;
    }

    void step(double dt, Inputs const& inputs_) {
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
        states.V_C_dc = outputs.V_C_dc;
    }

    struct Components {
        double L_a = -1;
        double L_b = -1;
        double L_c = -1;
        double C_dc = -1;
        double R_a = -1;
        double R_b = -1;
        double R_c = -1;
        double R_dc = -1;
        double R_load = -1;

        bool operator==(Components const& other) const {
            return
                L_a == other.L_a &&
                L_b == other.L_b &&
                L_c == other.L_c &&
                C_dc == other.C_dc &&
                R_a == other.R_a &&
                R_b == other.R_b &&
                R_c == other.R_c &&
                R_dc == other.R_dc &&
                R_load == other.R_load;
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
            double V_C_dc;
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
            double V3_a;
            double V3_b;
            double V3_c;
            double V_C_dc;
            double Vdc_n;
            double Vdc_p;
        };
        Eigen::Vector<double, NUM_OUTPUTS> data;
    };

    union Switches {
        struct {
            uint32_t S_p_a : 1;
            uint32_t S_p_b : 1;
            uint32_t S_p_c : 1;
            uint32_t S_n_a : 1;
            uint32_t S_n_b : 1;
            uint32_t S_n_c : 1;
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
    double m_dt_resolution = 0;
    TimestepErrorCorrectionMode m_dt_correction_mode = TimestepErrorCorrectionMode::NONE;
    double m_dt_error_accumulator = 0;
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
