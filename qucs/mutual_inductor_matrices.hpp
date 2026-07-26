
#pragma once

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

#include "on_off_delay.hpp"
#include "integrator.hpp"
#include "rlc2ss.h"

#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/LU"

#include "nlohmann/json.hpp"

#include <assert.h>

class Model_mutual_inductor {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    struct Switches;
    struct StateSpaceMatrices;

    Model_mutual_inductor() {}
    Model_mutual_inductor(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 4;
    static inline constexpr size_t NUM_OUTPUTS = 10;
    static inline constexpr size_t NUM_STATES = 4;
    static inline constexpr size_t NUM_SWITCHES = 0;
    static inline constexpr size_t NUM_DIODES = 0;

    enum class TimestepErrorCorrectionMode {
        // Ignore error in timestep length that is not a multiple of timestep resolution. Use this if
        // e.g. resolution is 0.1e-6 and the variation in timestep length is a multiple of that and
        // there should not ever be any error although floating point roundings may cause some.
        NONE,
        // Round the used timestep to closest multiple of resolution and store the error to accumulator
        // so that the timestep length error will be corrected when accumulator becomes a multiple of the
        // timestep resolution.
        ACCUMULATE
    };

    void setTimestepResolution(double dt, TimestepErrorCorrectionMode mode) {
        m_solver.enableInverseMatrixCaching(true);
        m_dt_resolution = dt;
        m_dt_correction_mode = mode;
    }

    void step(double dt, Inputs const& inputs_);

    /// @brief Add stepwise saturation curve to inductor. The inductance is reduced when the current
    /// exceeds the breakpoints and increased when current goes below the breakpoints.
    /// @param inductor Pointer to inductor in component struct e.g. &circuit.components.L0
    /// @param current Current breakpoints in ascending order. First breakpoint must be 0.
    /// @param inductance Inductance values at the breakpoints.
    /// Example:
    /// currents    = {0,       100,        200,       300}
    /// inductances = {   100e-6,    75e-6,      50e-6,     25e-6}
    void addInductorSaturation(double* inductor, std::vector<double> current, std::vector<double> inductance);

    union Inputs {
        Inputs() { data.setZero(); }
        Inputs(const Inputs& other) { data = other.data; }
        struct {
            double V1;
            double V2;
            double V3;
            double VSRC1;
        };
        Eigen::Vector<double, NUM_INPUTS> data;
    };

    union Outputs {
        Outputs() { data.setZero(); }
        Outputs(const Outputs& other) { data = other.data; }
        struct {
            double I_L1;
            double I_L2;
            double I_L3;
            double I_R3;
            double I_R4;
            double I_V3;
            double N1;
            double N2;
            double N3;
            double V_Cf;
        };
        Eigen::Vector<double, NUM_OUTPUTS> data;
    };

    struct Switches {


        uint64_t all() const;
        double smallestDelay();
        void step(double dt);
    };

    struct Components {
        double Cf = 0.0001;
        double FSRC1 = -1.0;
        double K12 = -1;
        double K21 = -1;
        double K31 = -1;
        double L1 = 1.0;
        double L2 = 1.0;
        double L3 = 1.0;
        double R1 = 10.0;
        double R2 = 10.0;
        double R3 = 0.01;
        double R4 = 10.0;

        uint64_t hash() const;
        bool operator==(Components const& other) const;
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
            double I_L2;
            double I_L3;
            double V_Cf;
        };
        Eigen::Vector<double, NUM_STATES> data;
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
    Switches switches;

  private:
    std::optional<rlc2ss::ZeroCrossingEvent> checkZeroCrossingEvents(Outputs const& prev_outputs);
    void stepWithZeroCrossingDetection(double dt);
    void stepModel(double dt);
    void updateStateSpaceMatrices();

    Integrator<Eigen::Vector<double, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_INPUTS>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components _M_components_DO_NOT_TOUCH;
    Switches _M_switches_DO_NOT_TOUCH;
    double m_dt_resolution = 0;
    TimestepErrorCorrectionMode m_dt_correction_mode = TimestepErrorCorrectionMode::NONE;
    double m_dt_error_accumulator = 0;
    using ZeroCrossCallback = std::function<std::optional<rlc2ss::ZeroCrossingEvent>(Outputs const& prev_outputs, Outputs const& new_outputs)>;
    std::vector<ZeroCrossCallback> m_zero_crossing_callbacks;
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
