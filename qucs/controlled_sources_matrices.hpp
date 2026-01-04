
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

class Model_controlled_sources {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    struct Switches;
    struct StateSpaceMatrices;

    Model_controlled_sources() {}
    Model_controlled_sources(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 3;
    static inline constexpr size_t NUM_OUTPUTS = 4;
    static inline constexpr size_t NUM_STATES = 3;
    static inline constexpr size_t NUM_SWITCHES = 0;

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
            double V1;
            double VSRC4;
            double VSRC5;
        };
        Eigen::Vector<double, NUM_INPUTS> data;
    };

    union Outputs {
        Outputs() { data.setZero(); }
        Outputs(const Outputs& other) { data = other.data; }
        struct {
            double I_C_2;
            double I_L1;
            double V_C_1;
            double V_C_2;
        };
        Eigen::Vector<double, NUM_OUTPUTS> data;
    };

    struct Switches {


        uint64_t all() const;
        double smallestDelay();
        void step(double dt);
    };

    struct Components {
        double C_1 = 9.999999999999999e-05;
        double C_2 = 0.0001;
        double ESRC3 = -1.0;
        double FSRC5 = -1.0;
        double GSRC1 = -1.0;
        double HSRC4 = -1.0;
        double L1 = 0.001;
        double R1 = 1.0;
        double R2 = 1.0;
        double R3 = 1.0;
        double R4 = 1.0;
        double R5 = 1.0;
        double R6 = 1000.0;
        double R7 = 1000.0;

        bool operator==(Components const& other) const {
            return
                C_1 == other.C_1 &&
                C_2 == other.C_2 &&
                ESRC3 == other.ESRC3 &&
                FSRC5 == other.FSRC5 &&
                GSRC1 == other.GSRC1 &&
                HSRC4 == other.HSRC4 &&
                L1 == other.L1 &&
                R1 == other.R1 &&
                R2 == other.R2 &&
                R3 == other.R3 &&
                R4 == other.R4 &&
                R5 == other.R5 &&
                R6 == other.R6 &&
                R7 == other.R7;
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
            double I_L1;
            double V_C_1;
            double V_C_2;
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
