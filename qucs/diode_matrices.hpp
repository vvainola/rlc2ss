
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
#include <unordered_map>

class Model_diode {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    struct Switches;
    struct StateSpaceMatrices;

    Model_diode() {}
    Model_diode(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 4;
    static inline constexpr size_t NUM_OUTPUTS = 11;
    static inline constexpr size_t NUM_STATES = 3;
    static inline constexpr size_t NUM_SWITCHES = 4;

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
            double V_D1;
            double V_D2;
            double V_D3;
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
            double I_R_D1;
            double I_R_D2;
            double I_R_D3;
            double N_D2_P;
            double N_D3_N;
            double _net1;
            double _net2;
            double _net4;
        };
        Eigen::Vector<double, NUM_OUTPUTS> data;
    };

    struct Switches {
        rlc2ss::OnOffDelay S1;
        rlc2ss::OnOffDelay S_D1;
        rlc2ss::OnOffDelay S_D2;
        rlc2ss::OnOffDelay S_D3;

        uint64_t all() const;
        double smallestDelay();
        void step(double dt);
    };

    struct Components {
        double K1 = 0.5;
        double L1 = 0.02;
        double L2 = 0.01;
        double L3 = 0.02;
        double R1 = 0.1;
        double R2 = 1.0;
        double R3 = 1.0;
        double R4 = 1.0;
        double R5 = 0.001;
        double R_D1 = -1.0;
        double R_D2 = -1.0;
        double R_D3 = -1.0;

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
    void resolveDiodeContinuity();
    Outputs calcInstantaneousOutputs(uint64_t switch_combination);
    StateSpaceMatrices const& calcStateSpaceMatrices(uint64_t switch_combination);
    uint64_t controlledSwitchMask() const;
    uint64_t closedDiodeMask() const;
    uint64_t inductorCurrentSignMask() const;
    uint64_t switchMaskWithClosedDiodes(uint64_t base_switch_mask, uint64_t closed_diode_mask) const;
    bool diodeClosed(size_t diode_idx) const;
    bool diodeControlledClosed(size_t diode_idx, uint64_t controlled_switch_mask) const;
    double diodeCurrent(size_t diode_idx, Outputs const& outputs_) const;
    double diodeForwardOverdrive(size_t diode_idx, Outputs const& outputs_) const;
    double inductorCurrentDiscontinuity(Outputs const& outputs_) const;
    void forceClosedDiodeMask(uint64_t closed_diode_mask);
    void releaseReverseCurrentDiodes();
    void stepWithZeroCrossingDetection(double dt);
    void stepModel(double dt);

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
    uint64_t m_last_continuity_switch_mask = ~uint64_t{0};
    uint64_t m_last_switch_mask = 0;
    std::unordered_map<uint64_t, uint64_t> m_diode_continuity_cache;
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
