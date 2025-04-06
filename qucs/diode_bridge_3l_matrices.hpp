
#pragma once

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include "integrator.hpp"
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

    static inline constexpr size_t NUM_INPUTS = 10;
    static inline constexpr size_t NUM_OUTPUTS = 31;
    static inline constexpr size_t NUM_STATES = 19;
    static inline constexpr size_t NUM_SWITCHES = 9;

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
        Inputs() {
            data.setZero();
        }
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
            double I_R_D_n_a;
            double I_R_D_n_b;
            double I_R_D_n_c;
            double I_R_D_p_a;
            double I_R_D_p_b;
            double I_R_D_p_c;
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
            uint64_t S_0_a : 1;
            uint64_t S_0_b : 1;
            uint64_t S_0_c : 1;
            uint64_t S_D_n_a : 1;
            uint64_t S_D_n_b : 1;
            uint64_t S_D_n_c : 1;
            uint64_t S_D_p_a : 1;
            uint64_t S_D_p_b : 1;
            uint64_t S_D_p_c : 1;
        };
        uint64_t all;
    };

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
        double R_D_n_a = -1;
        double R_D_n_b = -1;
        double R_D_n_c = -1;
        double R_D_p_a = -1;
        double R_D_p_b = -1;
        double R_D_p_c = -1;

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
                R_src_c == other.R_src_c &&
                R_D_n_a == other.R_D_n_a &&
                R_D_n_b == other.R_D_n_b &&
                R_D_n_c == other.R_D_n_c &&
                R_D_p_a == other.R_D_p_a &&
                R_D_p_b == other.R_D_p_b &&
                R_D_p_c == other.R_D_p_c;
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
    Switches switches = {.all = 0};

  private:
    void stepInternal(double dt);
    void updateStateSpaceMatrices();

    Integrator<Eigen::Vector<double, NUM_STATES>,
               Eigen::Matrix<double, NUM_STATES, NUM_STATES>>
        m_solver;
    StateSpaceMatrices m_ss;
    Components m_components_DO_NOT_TOUCH;
    Switches m_switches_DO_NOT_TOUCH = {.all = 0};
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
