
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

class Model_diode_bridge_3l {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    struct Switches;
    struct StateSpaceMatrices;

    Model_diode_bridge_3l() {}
    Model_diode_bridge_3l(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 10;
    static inline constexpr size_t NUM_OUTPUTS = 35;
    static inline constexpr size_t NUM_STATES = 19;
    static inline constexpr size_t NUM_SWITCHES = 15;
    static inline constexpr size_t NUM_DIODES = 6;
    static_assert(NUM_SWITCHES < 64,
                  "Generated models support at most 63 switches");

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
            double V_D_n_a;
            double V_D_n_b;
            double V_D_n_c;
            double V_D_p_a;
            double V_D_p_b;
            double V_D_p_c;
            double V_dc_src;
            double V_src_a;
            double V_src_b;
            double V_src_c;
        };
        Eigen::Vector<double, NUM_INPUTS> data;
    };

    union Outputs {
        Outputs() { data.setZero(); }
        Outputs(const Outputs& other) { data = other.data; }
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
            double I_R_D_n_a;
            double I_R_D_n_b;
            double I_R_D_n_c;
            double I_R_D_p_a;
            double I_R_D_p_b;
            double I_R_D_p_c;
            double N_cap_0;
            double N_cap_a;
            double N_cap_b;
            double N_cap_c;
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

    struct Switches {
        rlc2ss::OnOffDelay S_0_a;
        rlc2ss::OnOffDelay S_0_b;
        rlc2ss::OnOffDelay S_0_c;
        rlc2ss::OnOffDelay S_D_n_a;
        rlc2ss::OnOffDelay S_D_n_b;
        rlc2ss::OnOffDelay S_D_n_c;
        rlc2ss::OnOffDelay S_D_p_a;
        rlc2ss::OnOffDelay S_D_p_b;
        rlc2ss::OnOffDelay S_D_p_c;
        rlc2ss::OnOffDelay S_n_a;
        rlc2ss::OnOffDelay S_n_b;
        rlc2ss::OnOffDelay S_n_c;
        rlc2ss::OnOffDelay S_p_a;
        rlc2ss::OnOffDelay S_p_b;
        rlc2ss::OnOffDelay S_p_c;

        uint64_t all() const;
        double smallestDelay();
        void step(double dt);
    };

    struct Components {
        double C_dc_n1 = 0.01;
        double C_dc_n2 = 0.01;
        double C_dc_p1 = 0.01;
        double C_dc_p2 = 0.01;
        double C_f_a = 0.001;
        double C_f_b = 0.001;
        double C_f_c = 0.001;
        double L_conv_a = 1e-06;
        double L_conv_b = 1e-06;
        double L_conv_c = 1e-06;
        double L_dc_n = 1e-06;
        double L_dc_p = 1e-06;
        double L_dc_src = 1e-05;
        double L_grid_a = 1e-06;
        double L_grid_b = 1e-06;
        double L_grid_c = 1e-06;
        double L_src_a = 1e-06;
        double L_src_b = 1e-06;
        double L_src_c = 1e-06;
        double R_D_n_a = -1.0;
        double R_D_n_b = -1.0;
        double R_D_n_c = -1.0;
        double R_D_p_a = -1.0;
        double R_D_p_b = -1.0;
        double R_D_p_c = -1.0;
        double R_conv_a = 0.001;
        double R_conv_b = 0.001;
        double R_conv_c = 0.001;
        double R_dc_pn1 = 1000.0;
        double R_dc_pn2 = 1000.0;
        double R_dc_pp1 = 1000.0;
        double R_dc_pp2 = 1000.0;
        double R_dc_sn1 = 0.001;
        double R_dc_sn2 = 0.001;
        double R_dc_sp1 = 0.001;
        double R_dc_sp2 = 0.001;
        double R_dc_src_p = 1000.0;
        double R_dc_src_s = 1.0;
        double R_f_a = 0.001;
        double R_f_b = 0.001;
        double R_f_c = 0.001;
        double R_grid_a = 0.001;
        double R_grid_b = 0.001;
        double R_grid_c = 0.001;
        double R_src_a = 0.001;
        double R_src_b = 0.001;
        double R_src_c = 0.001;

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
    bool m_backward_euler_pending = false;
    uint64_t m_last_external_closed_switch_mask = ~uint64_t{0};
    uint64_t m_last_switch_mask = 0;
    using ZeroCrossCallback = std::function<std::optional<rlc2ss::ZeroCrossingEvent>(Outputs const& prev_outputs, Outputs const& new_outputs)>;
    std::vector<ZeroCrossCallback> m_zero_crossing_callbacks;
    static_assert(sizeof(double) * NUM_STATES == sizeof(States));
    static_assert(sizeof(double) * NUM_INPUTS == sizeof(Inputs));
    static_assert(sizeof(double) * NUM_OUTPUTS == sizeof(Outputs));
};

#pragma warning(default : 4127) // conditional expression is constant
#pragma warning(default : 4189) // local variable is initialized but not referenced
#pragma warning(default : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(default : 4408) // anonymous struct did not declare any data members
#pragma warning(default : 5054) // operator '&': deprecated between enumerations of different types
