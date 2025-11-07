
#pragma once

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

#include "on_off_delay.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include "integrator.hpp"
#include "nlohmann/json.hpp"
#include <assert.h>

class Model_diode_bridge {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    struct Switches;
    struct StateSpaceMatrices;

    Model_diode_bridge() {}
    Model_diode_bridge(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 9;
    static inline constexpr size_t NUM_OUTPUTS = 15;
    static inline constexpr size_t NUM_STATES = 4;
    static inline constexpr size_t NUM_SWITCHES = 6;

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

    void step(double dt, Inputs const& inputs_);

    union Inputs {
        Inputs() { data.setZero(); }
        Inputs(const Inputs& other) { data = other.data; }
        struct {
            double V_D_n_a;
            double V_D_n_b;
            double V_D_n_c;
            double V_D_p_a;
            double V_D_p_b;
            double V_D_p_c;
            double V_a;
            double V_b;
            double V_c;
        };
        Eigen::Vector<double, NUM_INPUTS> data;
    };

    union Outputs {
        Outputs() { data.setZero(); }
        Outputs(const Outputs& other) { data = other.data; }
        struct {
            double I_L_a;
            double I_L_b;
            double I_L_c;
            double I_R_D_n_a;
            double I_R_D_n_b;
            double I_R_D_n_c;
            double I_R_D_p_a;
            double I_R_D_p_b;
            double I_R_D_p_c;
            double N_dc_n;
            double N_dc_p;
            double V_C_dc;
            double _net0;
            double _net1;
            double _net2;
        };
        Eigen::Vector<double, NUM_OUTPUTS> data;
    };

    struct Switches {
        rlc2ss::OnOffDelay S_D_n_a;
        rlc2ss::OnOffDelay S_D_n_b;
        rlc2ss::OnOffDelay S_D_n_c;
        rlc2ss::OnOffDelay S_D_p_a;
        rlc2ss::OnOffDelay S_D_p_b;
        rlc2ss::OnOffDelay S_D_p_c;

        uint64_t all() const;
        double smallestDelay();
        void step(double dt);
    };

    struct Components {
        double C_dc = -1.0;
        double L_a = 1e-09;
        double L_b = 1e-09;
        double L_c = 1e-09;
        double R_D_n_a = -1.0;
        double R_D_n_b = -1.0;
        double R_D_n_c = -1.0;
        double R_D_p_a = -1.0;
        double R_D_p_b = -1.0;
        double R_D_p_c = -1.0;
        double R_a = 50.0;
        double R_b = 50.0;
        double R_c = 50.0;
        double R_dc = 50.0;
        double R_load = 50.0;

        bool operator==(Components const& other) const {
            return
                C_dc == other.C_dc &&
                L_a == other.L_a &&
                L_b == other.L_b &&
                L_c == other.L_c &&
                R_D_n_a == other.R_D_n_a &&
                R_D_n_b == other.R_D_n_b &&
                R_D_n_c == other.R_D_n_c &&
                R_D_p_a == other.R_D_p_a &&
                R_D_p_b == other.R_D_p_b &&
                R_D_p_c == other.R_D_p_c &&
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
        States(const States& other) {
            data = other.data;
        }
        struct {
            double I_L_a;
            double I_L_b;
            double I_L_c;
            double V_C_dc;
        };
        Eigen::Vector<double, NUM_STATES> data;
    };

    struct StateSpaceMatrices {
        Eigen::Matrix<double, NUM_STATES, NUM_STATES> A;
        Eigen::Matrix<double, NUM_STATES, NUM_INPUTS> B;
        Eigen::Matrix<double, NUM_OUTPUTS, NUM_STATES> C;
        Eigen::Matrix<double, NUM_OUTPUTS, NUM_INPUTS> D;
    };

    Eigen::Vector<double, NUM_STATES> dxdt(Eigen::Vector<double, NUM_STATES> const& state, double /*t*/) const {
        return m_ss.A * state + m_Bu;
    }

    Components components;
    Inputs inputs;
    States states;
    Outputs outputs;
    Switches switches;

  private:
    void stepWithZeroCrossingDetection(double dt);
    void stepModel(double dt);
    void updateStateSpaceMatrices();

    Integrator<Eigen::Vector<double, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components _M_components_DO_NOT_TOUCH;
    Switches _M_switches_DO_NOT_TOUCH;
    Eigen::Vector<double, NUM_STATES> m_Bu; // Bu term in "dxdt = Ax + Bu"
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
