
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

class Model_converter {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    struct Switches;
    struct StateSpaceMatrices;

    Model_converter() {}
    Model_converter(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 10;
    static inline constexpr size_t NUM_OUTPUTS = 28;
    static inline constexpr size_t NUM_STATES = 11;
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
            double V_D_a_n;
            double V_D_a_p;
            double V_D_b_n;
            double V_D_b_p;
            double V_D_c_n;
            double V_D_c_p;
            double V_a;
            double V_b;
            double V_c;
            double V_dc;
        };
        Eigen::Vector<double, NUM_INPUTS> data;
    };

    union Outputs {
        Outputs() { data.setZero(); }
        Outputs(const Outputs& other) { data = other.data; }
        struct {
            double I_C_a;
            double I_C_b;
            double I_C_c;
            double I_C_n;
            double I_C_p;
            double I_L_a;
            double I_L_b;
            double I_L_c;
            double I_L_g_a;
            double I_L_g_b;
            double I_L_g_c;
            double I_R_D_a_n;
            double I_R_D_a_p;
            double I_R_D_b_n;
            double I_R_D_b_p;
            double I_R_D_c_n;
            double I_R_D_c_p;
            double I_R_dc;
            double N_c_a;
            double N_c_b;
            double N_c_c;
            double N_dc_n;
            double N_dc_p;
            double V_C_a;
            double V_C_b;
            double V_C_c;
            double V_C_n;
            double V_C_p;
        };
        Eigen::Vector<double, NUM_OUTPUTS> data;
    };

    struct Switches {
        rlc2ss::OnOffDelay S_D_a_n;
        rlc2ss::OnOffDelay S_D_a_p;
        rlc2ss::OnOffDelay S_D_b_n;
        rlc2ss::OnOffDelay S_D_b_p;
        rlc2ss::OnOffDelay S_D_c_n;
        rlc2ss::OnOffDelay S_D_c_p;

        uint64_t all() const;
        double smallestDelay();
        void step(double dt);
    };

    struct Components {
        double C_a = 0.01;
        double C_b = 0.01;
        double C_c = 0.01;
        double C_n = 0.01;
        double C_p = 0.01;
        double L_a = -1.0;
        double L_b = -1.0;
        double L_c = -1.0;
        double L_g_a = -1.0;
        double L_g_b = -1.0;
        double L_g_c = -1.0;
        double R_D_a_n = -1.0;
        double R_D_a_p = -1.0;
        double R_D_b_n = -1.0;
        double R_D_b_p = -1.0;
        double R_D_c_n = -1.0;
        double R_D_c_p = -1.0;
        double R_a = -1.0;
        double R_b = -1.0;
        double R_c = -1.0;
        double R_dc = 1.0;
        double R_g_a = -1.0;
        double R_g_b = -1.0;
        double R_g_c = -1.0;
        double R_n_p = -1.0;
        double R_n_s = -1.0;
        double R_p_p = -1.0;
        double R_p_s = -1.0;

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
            double I_L_a;
            double I_L_b;
            double I_L_c;
            double I_L_g_a;
            double I_L_g_b;
            double I_L_g_c;
            double V_C_a;
            double V_C_b;
            double V_C_c;
            double V_C_n;
            double V_C_p;
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
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components _M_components_DO_NOT_TOUCH;
    Switches _M_switches_DO_NOT_TOUCH;
    Eigen::Vector<double, NUM_STATES> m_Bu; // Bu term in "dxdt = Ax + Bu"
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
