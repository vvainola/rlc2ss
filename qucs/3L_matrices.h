
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include "integrator.h"
#include <assert.h>

class Model_3L {
  public:
    struct Components;
    union Inputs;
    union Outputs;
    union States;
    union Switches;
    struct StateSpaceMatrices;
    StateSpaceMatrices calculateStateSpace(Components const& components, Switches switches);

    Model_3L(Components const& c);

    static inline constexpr size_t NUM_INPUTS = 4;
    static inline constexpr size_t NUM_OUTPUTS = 22;
    static inline constexpr size_t NUM_STATES = 15;
    static inline constexpr size_t NUM_SWITCHES = 9;

    Eigen::Vector<double, NUM_STATES> dxdt(Eigen::Vector<double, NUM_STATES> const& state, double /*t*/) const {
        return m_ss.A * state + m_Bu;
    }

    Eigen::Matrix<double, NUM_STATES, NUM_STATES> const& jacobian(const Eigen::Vector<double, NUM_STATES>& /*state*/, const double& /*t*/) const {
        return m_ss.A;
    }

    void step(double dt, Inputs const& inputs) {
        m_inputs.data = inputs.data;
        // Update state-space matrices if needed
        if (components != m_components_DO_NOT_TOUCH || switches.all != m_switches_DO_NOT_TOUCH.all) {
            m_components_DO_NOT_TOUCH = components;
            m_switches_DO_NOT_TOUCH.all = switches.all;
            m_ss = calculateStateSpace(components, switches);
            // Solve one step with backward euler to reduce numerical oscillations
            m_Bu = m_ss.B * m_inputs.data;
            states.data = m_solver.step_backward_euler(*this, states.data, 0.0, dt);

            // Update coefficients to make following steps with Tustin
            m_solver.update_tustin_coeffs(jacobian(states.data, dt), dt);
        } else {
            if (dt != m_dt_prev) {
                m_solver.update_tustin_coeffs(jacobian(states.data, dt), dt);
            }

            // Solve with Tustin for better accuracy
            m_Bu = m_ss.B * m_inputs.data;
            states.data = m_solver.step_tustin_fast(*this, states.data, 0.0, dt);
        }
        m_dt_prev = dt;


        // Update output
        outputs.data = m_ss.C * states.data + m_ss.D * m_inputs.data;

        // Update states from outputs to have correct values for dependent states
        states.I_L_conv_a = outputs.I_L_conv_a;
        states.I_L_conv_b = outputs.I_L_conv_b;
        states.I_L_conv_c = outputs.I_L_conv_c;
        states.I_L_dc_src = outputs.I_L_dc_src;
        states.I_L_grid_a = outputs.I_L_grid_a;
        states.I_L_grid_b = outputs.I_L_grid_b;
        states.I_L_grid_c = outputs.I_L_grid_c;
        states.I_L_src_a = outputs.I_L_src_a;
        states.I_L_src_b = outputs.I_L_src_b;
        states.I_L_src_c = outputs.I_L_src_c;
        states.V_C_dc_n = outputs.V_C_dc_n;
        states.V_C_dc_p = outputs.V_C_dc_p;
        states.V_C_f_a = outputs.V_C_f_a;
        states.V_C_f_b = outputs.V_C_f_b;
        states.V_C_f_c = outputs.V_C_f_c;
    }

    struct Components {
        double L_conv_a = -1;
        double L_conv_b = -1;
        double L_conv_c = -1;
        double L_dc_src = -1;
        double L_grid_a = -1;
        double L_grid_b = -1;
        double L_grid_c = -1;
        double L_src_a = -1;
        double L_src_b = -1;
        double L_src_c = -1;
        double C_dc_n = -1;
        double C_dc_p = -1;
        double C_f_a = -1;
        double C_f_b = -1;
        double C_f_c = -1;
        double R_conv_a = -1;
        double R_conv_b = -1;
        double R_conv_c = -1;
        double R_dc_n = -1;
        double R_dc_p = -1;
        double R_dc_src = -1;
        double R_f_a = -1;
        double R_f_b = -1;
        double R_f_c = -1;
        double R_grid_a = -1;
        double R_grid_b = -1;
        double R_grid_c = -1;
        double R_src_a = -1;
        double R_src_b = -1;
        double R_src_c = -1;
        double R_dc_n_0 = -1;

        bool operator==(Components const& other) const {
            return 
                L_conv_a == other.L_conv_a &&
                L_conv_b == other.L_conv_b &&
                L_conv_c == other.L_conv_c &&
                L_dc_src == other.L_dc_src &&
                L_grid_a == other.L_grid_a &&
                L_grid_b == other.L_grid_b &&
                L_grid_c == other.L_grid_c &&
                L_src_a == other.L_src_a &&
                L_src_b == other.L_src_b &&
                L_src_c == other.L_src_c &&
                C_dc_n == other.C_dc_n &&
                C_dc_p == other.C_dc_p &&
                C_f_a == other.C_f_a &&
                C_f_b == other.C_f_b &&
                C_f_c == other.C_f_c &&
                R_conv_a == other.R_conv_a &&
                R_conv_b == other.R_conv_b &&
                R_conv_c == other.R_conv_c &&
                R_dc_n == other.R_dc_n &&
                R_dc_p == other.R_dc_p &&
                R_dc_src == other.R_dc_src &&
                R_f_a == other.R_f_a &&
                R_f_b == other.R_f_b &&
                R_f_c == other.R_f_c &&
                R_grid_a == other.R_grid_a &&
                R_grid_b == other.R_grid_b &&
                R_grid_c == other.R_grid_c &&
                R_src_a == other.R_src_a &&
                R_src_b == other.R_src_b &&
                R_src_c == other.R_src_c &&
                R_dc_n_0 == other.R_dc_n_0;
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
            double I_L_dc_src;
            double I_L_grid_a;
            double I_L_grid_b;
            double I_L_grid_c;
            double I_L_src_a;
            double I_L_src_b;
            double I_L_src_c;
            double V_C_dc_n;
            double V_C_dc_p;
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
            double I_L_dc_src;
            double I_L_grid_a;
            double I_L_grid_b;
            double I_L_grid_c;
            double I_L_src_a;
            double I_L_src_b;
            double I_L_src_c;
            double I_R_dc_n_0;
            double N_conv_a;
            double N_conv_b;
            double N_conv_c;
            double N_dc_0;
            double N_dc_n;
            double N_dc_p;
            double V_C_dc_n;
            double V_C_dc_p;
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

    static_assert(sizeof(double) * NUM_STATES == sizeof(States));
    static_assert(sizeof(double) * NUM_INPUTS == sizeof(Inputs));
    static_assert(sizeof(double) * NUM_OUTPUTS == sizeof(Outputs));
};

Model_3L::Model_3L(Components const& c)
    : components(c),
      m_components_DO_NOT_TOUCH(c) {
    m_ss = calculateStateSpace(components, switches);
    assert(components.L_conv_a != -1);
    assert(components.L_conv_b != -1);
    assert(components.L_conv_c != -1);
    assert(components.L_dc_src != -1);
    assert(components.L_grid_a != -1);
    assert(components.L_grid_b != -1);
    assert(components.L_grid_c != -1);
    assert(components.L_src_a != -1);
    assert(components.L_src_b != -1);
    assert(components.L_src_c != -1);
    assert(components.C_dc_n != -1);
    assert(components.C_dc_p != -1);
    assert(components.C_f_a != -1);
    assert(components.C_f_b != -1);
    assert(components.C_f_c != -1);
    assert(components.R_conv_a != -1);
    assert(components.R_conv_b != -1);
    assert(components.R_conv_c != -1);
    assert(components.R_dc_n != -1);
    assert(components.R_dc_p != -1);
    assert(components.R_dc_src != -1);
    assert(components.R_f_a != -1);
    assert(components.R_f_b != -1);
    assert(components.R_f_c != -1);
    assert(components.R_grid_a != -1);
    assert(components.R_grid_b != -1);
    assert(components.R_grid_c != -1);
    assert(components.R_src_a != -1);
    assert(components.R_src_b != -1);
    assert(components.R_src_c != -1);
    assert(components.R_dc_n_0 != -1);
}

Model_3L::StateSpaceMatrices calculateStateSpace_0(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_1(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_2(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_3(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_4(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_5(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_6(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_7(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_8(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_10(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_12(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_14(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_16(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_17(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_20(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_21(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_24(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_28(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_32(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_33(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_34(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_35(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_40(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_42(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_48(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_49(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_56(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_64(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_66(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_68(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_70(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_80(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_84(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_96(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_98(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_112(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_128(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_129(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_132(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_133(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_136(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_140(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_160(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_161(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_168(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_192(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_196(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_224(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_256(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_257(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_258(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_259(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_264(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_266(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_272(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_273(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_280(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_320(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_322(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_336(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_384(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_385(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_392(Model_3L::Components const& c);
Model_3L::StateSpaceMatrices calculateStateSpace_448(Model_3L::Components const& c);

struct Topology {
    Model_3L::Components components;
    Model_3L::Switches switches;
    Model_3L::StateSpaceMatrices state_space;
};
    
Model_3L::StateSpaceMatrices Model_3L::calculateStateSpace(Model_3L::Components const& components, Model_3L::Switches switches)
{
    static std::vector<Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Topology const& t) {
        return t.components == components && t.switches.all == switches.all;
    });
    if (it != state_space_cache.end()) {
        return it->state_space;
    }
    Model_3L::StateSpaceMatrices state_space;

    switch (switches.all) {

		case 0: state_space = calculateStateSpace_0(components); break;
		case 1: state_space = calculateStateSpace_1(components); break;
		case 2: state_space = calculateStateSpace_2(components); break;
		case 3: state_space = calculateStateSpace_3(components); break;
		case 4: state_space = calculateStateSpace_4(components); break;
		case 5: state_space = calculateStateSpace_5(components); break;
		case 6: state_space = calculateStateSpace_6(components); break;
		case 7: state_space = calculateStateSpace_7(components); break;
		case 8: state_space = calculateStateSpace_8(components); break;
		case 10: state_space = calculateStateSpace_10(components); break;
		case 12: state_space = calculateStateSpace_12(components); break;
		case 14: state_space = calculateStateSpace_14(components); break;
		case 16: state_space = calculateStateSpace_16(components); break;
		case 17: state_space = calculateStateSpace_17(components); break;
		case 20: state_space = calculateStateSpace_20(components); break;
		case 21: state_space = calculateStateSpace_21(components); break;
		case 24: state_space = calculateStateSpace_24(components); break;
		case 28: state_space = calculateStateSpace_28(components); break;
		case 32: state_space = calculateStateSpace_32(components); break;
		case 33: state_space = calculateStateSpace_33(components); break;
		case 34: state_space = calculateStateSpace_34(components); break;
		case 35: state_space = calculateStateSpace_35(components); break;
		case 40: state_space = calculateStateSpace_40(components); break;
		case 42: state_space = calculateStateSpace_42(components); break;
		case 48: state_space = calculateStateSpace_48(components); break;
		case 49: state_space = calculateStateSpace_49(components); break;
		case 56: state_space = calculateStateSpace_56(components); break;
		case 64: state_space = calculateStateSpace_64(components); break;
		case 66: state_space = calculateStateSpace_66(components); break;
		case 68: state_space = calculateStateSpace_68(components); break;
		case 70: state_space = calculateStateSpace_70(components); break;
		case 80: state_space = calculateStateSpace_80(components); break;
		case 84: state_space = calculateStateSpace_84(components); break;
		case 96: state_space = calculateStateSpace_96(components); break;
		case 98: state_space = calculateStateSpace_98(components); break;
		case 112: state_space = calculateStateSpace_112(components); break;
		case 128: state_space = calculateStateSpace_128(components); break;
		case 129: state_space = calculateStateSpace_129(components); break;
		case 132: state_space = calculateStateSpace_132(components); break;
		case 133: state_space = calculateStateSpace_133(components); break;
		case 136: state_space = calculateStateSpace_136(components); break;
		case 140: state_space = calculateStateSpace_140(components); break;
		case 160: state_space = calculateStateSpace_160(components); break;
		case 161: state_space = calculateStateSpace_161(components); break;
		case 168: state_space = calculateStateSpace_168(components); break;
		case 192: state_space = calculateStateSpace_192(components); break;
		case 196: state_space = calculateStateSpace_196(components); break;
		case 224: state_space = calculateStateSpace_224(components); break;
		case 256: state_space = calculateStateSpace_256(components); break;
		case 257: state_space = calculateStateSpace_257(components); break;
		case 258: state_space = calculateStateSpace_258(components); break;
		case 259: state_space = calculateStateSpace_259(components); break;
		case 264: state_space = calculateStateSpace_264(components); break;
		case 266: state_space = calculateStateSpace_266(components); break;
		case 272: state_space = calculateStateSpace_272(components); break;
		case 273: state_space = calculateStateSpace_273(components); break;
		case 280: state_space = calculateStateSpace_280(components); break;
		case 320: state_space = calculateStateSpace_320(components); break;
		case 322: state_space = calculateStateSpace_322(components); break;
		case 336: state_space = calculateStateSpace_336(components); break;
		case 384: state_space = calculateStateSpace_384(components); break;
		case 385: state_space = calculateStateSpace_385(components); break;
		case 392: state_space = calculateStateSpace_392(components); break;
		case 448: state_space = calculateStateSpace_448(components); break;
    default:
        assert(0);
    }
    state_space_cache.push_back(Topology{
        .components = components,
        .switches = switches,
        .state_space = state_space});

    return state_space;
}

Model_3L::StateSpaceMatrices calculateStateSpace_0(Model_3L::Components const& c) // 
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, L_grid_a + L_grid_b + L_src_a + L_src_b, L_grid_a + L_src_a, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, L_grid_a + L_src_a, L_grid_a + L_grid_c + L_src_a + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_f_a - R_f_b - R_grid_a - R_grid_b - R_src_a - R_src_b, -R_f_a - R_grid_a - R_src_a, 0, 0, -1, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_f_a - R_grid_a - R_src_a, -R_f_a - R_f_c - R_grid_a - R_grid_c - R_src_a - R_src_c, 0, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, -1, 1, 0,
			 0, -1, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, R_grid_a + R_src_a, R_grid_a + R_src_a, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_1(Model_3L::Components const& c) //  S_0_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, 0, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, 0, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_2(Model_3L::Components const& c) //  S_0_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, 0, 0, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b, 1, 0, 1, -1, 0,
			 0, 0, 0, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_grid_b - R_src_b, -R_conv_b - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b - R_f_c - R_grid_c - R_src_c, 1, 0, 0, -1, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_3(Model_3L::Components const& c) //  S_0_a S_0_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, 0, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, 0, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_4(Model_3L::Components const& c) //  S_0_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_c + L_grid_a + L_src_a, L_conv_c, L_conv_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_c, L_conv_c + L_grid_b + L_src_b, L_conv_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_c, L_conv_c, L_conv_c + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, 0, 0, R_dc_n, 0, 0, 0, -R_conv_c - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_a - R_src_a, -R_conv_c - R_dc_n - R_dc_n_0 - R_f_c, -R_conv_c - R_dc_n - R_dc_n_0, 1, 0, 1, 0, -1,
			 0, 0, 0, R_dc_n, 0, 0, 0, -R_conv_c - R_dc_n - R_dc_n_0 - R_f_c, -R_conv_c - R_dc_n - R_dc_n_0 - R_f_b - R_f_c - R_grid_b - R_src_b, -R_conv_c - R_dc_n - R_dc_n_0, 1, 0, 0, 1, -1,
			 0, 0, 0, R_dc_n, 0, 0, 0, -R_conv_c - R_dc_n - R_dc_n_0, -R_conv_c - R_dc_n - R_dc_n_0, -R_conv_c - R_dc_n - R_dc_n_0 - R_grid_c - R_src_c, 1, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_5(Model_3L::Components const& c) //  S_0_a S_0_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, 0, R_conv_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, 0, R_conv_a + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, 0, R_conv_a + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_6(Model_3L::Components const& c) //  S_0_b S_0_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_b + L_conv_c, 0, 0, 0, 0, -L_conv_b, -L_conv_b, -L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_b - R_conv_c - R_f_b - R_f_c, 0, 0, 0, 0, R_conv_b + R_f_b, R_conv_b, R_conv_b + R_f_b + R_f_c, 0, 0, 0, 1, -1,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b, 1, 0, 1, -1, 0,
			 0, 0, R_conv_b, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_grid_b - R_src_b, -R_conv_b - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b + R_f_c, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b - R_f_c - R_grid_c - R_src_c, 1, 0, 0, -1, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_7(Model_3L::Components const& c) //  S_0_a S_0_b S_0_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, -R_conv_a - R_f_a, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_8(Model_3L::Components const& c) //  S_n_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_10(Model_3L::Components const& c) //  S_0_b S_n_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, 0, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_f_a - R_f_b, 0, R_dc_n, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 1, 0, 1, -1, 0,
			 0, R_dc_n, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_12(Model_3L::Components const& c) //  S_0_c S_n_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_a - R_conv_c - R_dc_n - R_f_a - R_f_c, R_dc_n, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 1, 0, 1, 0, -1,
			 0, 0, R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, 0, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_14(Model_3L::Components const& c) //  S_0_b S_0_c S_n_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_f_a - R_f_b, -R_conv_a - R_dc_n - R_f_a, R_dc_n, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 1, 0, 1, -1, 0,
			 0, -R_conv_a - R_dc_n - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_f_a - R_f_c, R_dc_n, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 1, 0, 1, 0, -1,
			 0, R_dc_n, R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_dc_n, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, R_dc_n, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, R_dc_n, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_16(Model_3L::Components const& c) //  S_n_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_conv_b - R_dc_n_0 - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_f_b, 0, 0, 1, -1, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_grid_b - R_src_b, -R_conv_b - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_conv_b - R_dc_n_0 - R_f_b, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_f_b - R_f_c - R_grid_c - R_src_c, 0, 0, 0, -1, 1,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_17(Model_3L::Components const& c) //  S_0_a S_n_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, 0, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_f_a - R_f_b, 0, -R_dc_n, 0, 0, 0, R_conv_a + R_dc_n, R_conv_a + R_dc_n + R_f_a + R_f_b, R_conv_a + R_dc_n + R_f_a, -1, 0, 1, -1, 0,
			 0, -R_dc_n, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_n, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_f_a + R_f_b, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_dc_n + R_f_a, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_20(Model_3L::Components const& c) //  S_0_c S_n_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_b + L_conv_c, 0, 0, 0, 0, -L_conv_b, -L_conv_b, -L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_b - R_conv_c - R_dc_n - R_f_b - R_f_c, R_dc_n, 0, 0, 0, R_conv_b + R_f_b, R_conv_b, R_conv_b + R_f_b + R_f_c, 1, 0, 0, 1, -1,
			 0, 0, R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b, 0, 0, 0, 0, -R_conv_b - R_dc_n_0 - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_f_b, 0, 0, 1, -1, 0,
			 0, 0, R_conv_b, 0, 0, 0, 0, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_grid_b - R_src_b, -R_conv_b - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b + R_f_c, 0, 0, 0, 0, -R_conv_b - R_dc_n_0 - R_f_b, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_f_b - R_f_c - R_grid_c - R_src_c, 0, 0, 0, -1, 1,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_21(Model_3L::Components const& c) //  S_0_a S_0_c S_n_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_f_a - R_f_b, -R_conv_a - R_f_a, -R_dc_n, 0, 0, 0, R_conv_a + R_dc_n, R_conv_a + R_dc_n + R_f_a + R_f_b, R_conv_a + R_dc_n + R_f_a, -1, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, -R_dc_n, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_n, R_conv_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_f_a + R_f_b, R_conv_a + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_dc_n + R_f_a, R_conv_a + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_24(Model_3L::Components const& c) //  S_n_a S_n_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, 0, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, 0, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_28(Model_3L::Components const& c) //  S_0_c S_n_a S_n_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, -R_conv_a - R_f_a, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_f_a - R_f_c, R_dc_n, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 1, 0, 1, 0, -1,
			 0, 0, R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_32(Model_3L::Components const& c) //  S_n_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_c + L_grid_a + L_src_a, L_conv_c, L_conv_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_c, L_conv_c + L_grid_b + L_src_b, L_conv_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_c, L_conv_c, L_conv_c + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_conv_c - R_dc_n_0 - R_f_a - R_f_c - R_grid_a - R_src_a, -R_conv_c - R_dc_n_0 - R_f_c, -R_conv_c - R_dc_n_0, 0, 0, 1, 0, -1,
			 0, 0, 0, 0, 0, 0, 0, -R_conv_c - R_dc_n_0 - R_f_c, -R_conv_c - R_dc_n_0 - R_f_b - R_f_c - R_grid_b - R_src_b, -R_conv_c - R_dc_n_0, 0, 0, 0, 1, -1,
			 0, 0, 0, 0, 0, 0, 0, -R_conv_c - R_dc_n_0, -R_conv_c - R_dc_n_0, -R_conv_c - R_dc_n_0 - R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_33(Model_3L::Components const& c) //  S_0_a S_n_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_a - R_conv_c - R_dc_n - R_f_a - R_f_c, -R_dc_n, 0, 0, 0, R_conv_a + R_dc_n, R_conv_a + R_dc_n + R_f_a, R_conv_a + R_dc_n + R_f_a + R_f_c, -1, 0, 1, 0, -1,
			 0, 0, -R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, 0, R_conv_a + R_dc_n, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, 0, R_conv_a + R_dc_n + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, 0, R_conv_a + R_dc_n + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_34(Model_3L::Components const& c) //  S_0_b S_n_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_b + L_conv_c, 0, 0, 0, 0, -L_conv_b, -L_conv_b, -L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_b - R_conv_c - R_dc_n - R_f_b - R_f_c, -R_dc_n, 0, 0, 0, R_conv_b + R_dc_n + R_f_b, R_conv_b + R_dc_n, R_conv_b + R_dc_n + R_f_b + R_f_c, -1, 0, 0, 1, -1,
			 0, 0, -R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, 0, R_conv_b + R_dc_n + R_f_b, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b, 1, 0, 1, -1, 0,
			 0, 0, R_conv_b + R_dc_n, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_grid_b - R_src_b, -R_conv_b - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, 0, R_conv_b + R_dc_n + R_f_b + R_f_c, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b - R_f_c - R_grid_c - R_src_c, 1, 0, 0, -1, 1,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_35(Model_3L::Components const& c) //  S_0_a S_0_b S_n_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, -R_conv_a - R_f_a, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_f_a - R_f_c, -R_dc_n, 0, 0, 0, R_conv_a + R_dc_n, R_conv_a + R_dc_n + R_f_a, R_conv_a + R_dc_n + R_f_a + R_f_c, -1, 0, 1, 0, -1,
			 0, 0, -R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a + R_dc_n, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_dc_n + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_dc_n + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_40(Model_3L::Components const& c) //  S_n_a S_n_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, 0, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_42(Model_3L::Components const& c) //  S_0_b S_n_a S_n_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_f_a - R_f_b, -R_conv_a - R_f_a, R_dc_n, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 1, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, R_dc_n, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_48(Model_3L::Components const& c) //  S_n_b S_n_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_b + L_conv_c, 0, 0, 0, 0, -L_conv_b, -L_conv_b, -L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_b - R_conv_c - R_f_b - R_f_c, 0, 0, 0, 0, R_conv_b + R_f_b, R_conv_b, R_conv_b + R_f_b + R_f_c, 0, 0, 0, 1, -1,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b, 0, 0, 0, 0, -R_conv_b - R_dc_n_0 - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_f_b, 0, 0, 1, -1, 0,
			 0, 0, R_conv_b, 0, 0, 0, 0, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_grid_b - R_src_b, -R_conv_b - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b + R_f_c, 0, 0, 0, 0, -R_conv_b - R_dc_n_0 - R_f_b, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_f_b - R_f_c - R_grid_c - R_src_c, 0, 0, 0, -1, 1,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_49(Model_3L::Components const& c) //  S_0_a S_n_b S_n_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_f_a - R_f_b, -R_conv_a - R_dc_n - R_f_a, -R_dc_n, 0, 0, 0, R_conv_a + R_dc_n, R_conv_a + R_dc_n + R_f_a + R_f_b, R_conv_a + R_dc_n + R_f_a, -1, 0, 1, -1, 0,
			 0, -R_conv_a - R_dc_n - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_f_a - R_f_c, -R_dc_n, 0, 0, 0, R_conv_a + R_dc_n, R_conv_a + R_dc_n + R_f_a, R_conv_a + R_dc_n + R_f_a + R_f_c, -1, 0, 1, 0, -1,
			 0, -R_dc_n, -R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_n, R_conv_a + R_dc_n, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_f_a + R_f_b, R_conv_a + R_dc_n + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_dc_n + R_f_a, R_conv_a + R_dc_n + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, -R_dc_n, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, -R_dc_n, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_56(Model_3L::Components const& c) //  S_n_a S_n_b S_n_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, -R_conv_a - R_f_a, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_64(Model_3L::Components const& c) //  S_p_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, 0, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, 0, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, 0, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_66(Model_3L::Components const& c) //  S_0_b S_p_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, 0, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_p - R_f_a - R_f_b, 0, -R_dc_p, 0, 0, 0, R_conv_a + R_dc_p, R_conv_a + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_p + R_f_a, 0, -1, 1, -1, 0,
			 0, -R_dc_p, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_p, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_dc_p + R_f_a + R_f_b, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_dc_p + R_f_a, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_68(Model_3L::Components const& c) //  S_0_c S_p_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_a - R_conv_c - R_dc_p - R_f_a - R_f_c, -R_dc_p, 0, 0, 0, R_conv_a + R_dc_p, R_conv_a + R_dc_p + R_f_a, R_conv_a + R_dc_p + R_f_a + R_f_c, 0, -1, 1, 0, -1,
			 0, 0, -R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, 0, R_conv_a + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, 0, R_conv_a + R_dc_p + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, 0, R_conv_a + R_dc_p + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_70(Model_3L::Components const& c) //  S_0_b S_0_c S_p_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_dc_p - R_f_a, -R_dc_p, 0, 0, 0, R_conv_a + R_dc_p, R_conv_a + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_p + R_f_a, 0, -1, 1, -1, 0,
			 0, -R_conv_a - R_dc_p - R_f_a, -R_conv_a - R_conv_c - R_dc_p - R_f_a - R_f_c, -R_dc_p, 0, 0, 0, R_conv_a + R_dc_p, R_conv_a + R_dc_p + R_f_a, R_conv_a + R_dc_p + R_f_a + R_f_c, 0, -1, 1, 0, -1,
			 0, -R_dc_p, -R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_p, R_conv_a + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_p + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_dc_p + R_f_a, R_conv_a + R_dc_p + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_80(Model_3L::Components const& c) //  S_n_b S_p_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, 0, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_dc_p - R_f_a - R_f_b, 0, -R_dc_n - R_dc_p, 0, 0, 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_n + R_dc_p + R_f_a, -1, -1, 1, -1, 0,
			 0, -R_dc_n - R_dc_p, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_dc_p, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_b, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_dc_n + R_dc_p + R_f_a, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_84(Model_3L::Components const& c) //  S_0_c S_n_b S_p_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_dc_p - R_f_a, -R_dc_n - R_dc_p, 0, 0, 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_n + R_dc_p + R_f_a, -1, -1, 1, -1, 0,
			 0, -R_conv_a - R_dc_p - R_f_a, -R_conv_a - R_conv_c - R_dc_p - R_f_a - R_f_c, -R_dc_p, 0, 0, 0, R_conv_a + R_dc_p, R_conv_a + R_dc_p + R_f_a, R_conv_a + R_dc_p + R_f_a + R_f_c, 0, -1, 1, 0, -1,
			 0, -R_dc_n - R_dc_p, -R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_p + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_conv_a + R_dc_p + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_96(Model_3L::Components const& c) //  S_n_c S_p_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_a - R_conv_c - R_dc_n - R_dc_p - R_f_a - R_f_c, -R_dc_n - R_dc_p, 0, 0, 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_c, -1, -1, 1, 0, -1,
			 0, 0, -R_dc_n - R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, 0, R_conv_a + R_dc_n + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, 0, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, 0, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_98(Model_3L::Components const& c) //  S_0_b S_n_c S_p_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_dc_p - R_f_a, -R_dc_p, 0, 0, 0, R_conv_a + R_dc_p, R_conv_a + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_p + R_f_a, 0, -1, 1, -1, 0,
			 0, -R_conv_a - R_dc_p - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_dc_p - R_f_a - R_f_c, -R_dc_n - R_dc_p, 0, 0, 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_c, -1, -1, 1, 0, -1,
			 0, -R_dc_p, -R_dc_n - R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_p, R_conv_a + R_dc_n + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_dc_p + R_f_a, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_112(Model_3L::Components const& c) //  S_n_b S_n_c S_p_a
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_dc_n - R_dc_p - R_f_a, -R_dc_n - R_dc_p, 0, 0, 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_n + R_dc_p + R_f_a, -1, -1, 1, -1, 0,
			 0, -R_conv_a - R_dc_n - R_dc_p - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_dc_p - R_f_a - R_f_c, -R_dc_n - R_dc_p, 0, 0, 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_c, -1, -1, 1, 0, -1,
			 0, -R_dc_n - R_dc_p, -R_dc_n - R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_n + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, -R_dc_n, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_128(Model_3L::Components const& c) //  S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, 0, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b, 1, 1, 1, -1, 0,
			 0, 0, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_b - R_src_b, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, 0, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b - R_f_c - R_grid_c - R_src_c, 1, 1, 0, -1, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_129(Model_3L::Components const& c) //  S_0_a S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, 0, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_p - R_f_a - R_f_b, 0, R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 1, 1, -1, 0,
			 0, R_dc_p, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, 0, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_132(Model_3L::Components const& c) //  S_0_c S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_b + L_conv_c, 0, 0, 0, 0, -L_conv_b, -L_conv_b, -L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_b - R_conv_c - R_dc_p - R_f_b - R_f_c, -R_dc_p, 0, 0, 0, R_conv_b + R_dc_p + R_f_b, R_conv_b + R_dc_p, R_conv_b + R_dc_p + R_f_b + R_f_c, 0, -1, 0, 1, -1,
			 0, 0, -R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, 0, R_conv_b + R_dc_p + R_f_b, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b, 1, 1, 1, -1, 0,
			 0, 0, R_conv_b + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_b - R_src_b, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, 0, R_conv_b + R_dc_p + R_f_b + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b - R_f_c - R_grid_c - R_src_c, 1, 1, 0, -1, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_133(Model_3L::Components const& c) //  S_0_a S_0_c S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_f_a, R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 1, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, R_dc_p, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_136(Model_3L::Components const& c) //  S_n_a S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, 0, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_dc_p - R_f_a - R_f_b, 0, R_dc_n + R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 1, 1, 1, -1, 0,
			 0, R_dc_n + R_dc_p, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, 0, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_140(Model_3L::Components const& c) //  S_0_c S_n_a S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_dc_n - R_f_a, R_dc_n + R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 1, 1, 1, -1, 0,
			 0, -R_conv_a - R_dc_n - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_f_a - R_f_c, R_dc_n, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 1, 0, 1, 0, -1,
			 0, R_dc_n + R_dc_p, R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_dc_n, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, R_dc_n, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_160(Model_3L::Components const& c) //  S_n_c S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_b + L_conv_c, 0, 0, 0, 0, -L_conv_b, -L_conv_b, -L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_b - R_conv_c - R_dc_n - R_dc_p - R_f_b - R_f_c, -R_dc_n - R_dc_p, 0, 0, 0, R_conv_b + R_dc_n + R_dc_p + R_f_b, R_conv_b + R_dc_n + R_dc_p, R_conv_b + R_dc_n + R_dc_p + R_f_b + R_f_c, -1, -1, 0, 1, -1,
			 0, 0, -R_dc_n - R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, 0, R_conv_b + R_dc_n + R_dc_p + R_f_b, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b, 1, 1, 1, -1, 0,
			 0, 0, R_conv_b + R_dc_n + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_b - R_src_b, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, 0, R_conv_b + R_dc_n + R_dc_p + R_f_b + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b - R_f_c - R_grid_c - R_src_c, 1, 1, 0, -1, 1,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_161(Model_3L::Components const& c) //  S_0_a S_n_c S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_f_a, R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 1, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_f_a - R_f_c, -R_dc_n, 0, 0, 0, R_conv_a + R_dc_n, R_conv_a + R_dc_n + R_f_a, R_conv_a + R_dc_n + R_f_a + R_f_c, -1, 0, 1, 0, -1,
			 0, R_dc_p, -R_dc_n, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a + R_dc_n, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_dc_n + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_dc_n + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_168(Model_3L::Components const& c) //  S_n_a S_n_c S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_f_a, R_dc_n + R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 1, 1, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, R_dc_n + R_dc_p, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_192(Model_3L::Components const& c) //  S_p_a S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, 0, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, 0, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_c, 0, 0, 0, -L_grid_c, 0, 0, -L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, 0, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_f_a, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_c - R_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_196(Model_3L::Components const& c) //  S_0_c S_p_a S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, -R_conv_a - R_f_a, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_dc_p - R_f_a - R_f_c, -R_dc_p, 0, 0, 0, R_conv_a + R_dc_p, R_conv_a + R_dc_p + R_f_a, R_conv_a + R_dc_p + R_f_a + R_f_c, 0, -1, 1, 0, -1,
			 0, 0, -R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_dc_p + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_dc_p + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_224(Model_3L::Components const& c) //  S_n_c S_p_a S_p_b
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, -R_conv_a - R_f_a, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_dc_p - R_f_a - R_f_c, -R_dc_n - R_dc_p, 0, 0, 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_c, -1, -1, 1, 0, -1,
			 0, 0, -R_dc_n - R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a + R_dc_n + R_dc_p, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, -R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_256(Model_3L::Components const& c) //  S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_c + L_grid_a + L_src_a, L_conv_c, L_conv_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_c, L_conv_c + L_grid_b + L_src_b, L_conv_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, L_conv_c, L_conv_c, L_conv_c + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, 0, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_c - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_a - R_src_a, -R_conv_c - R_dc_n - R_dc_n_0 - R_dc_p - R_f_c, -R_conv_c - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 1, 0, -1,
			 0, 0, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_c - R_dc_n - R_dc_n_0 - R_dc_p - R_f_c, -R_conv_c - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b - R_f_c - R_grid_b - R_src_b, -R_conv_c - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 1, -1,
			 0, 0, 0, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_c - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_c - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_c - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_c - R_src_c, 1, 1, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_257(Model_3L::Components const& c) //  S_0_a S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_a - R_conv_c - R_dc_p - R_f_a - R_f_c, R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 1, 1, 0, -1,
			 0, 0, R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, 0, R_conv_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, 0, R_conv_a + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, 0, R_conv_a + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 1, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_258(Model_3L::Components const& c) //  S_0_b S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_b + L_conv_c, 0, 0, 0, 0, -L_conv_b, -L_conv_b, -L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_b - R_conv_c - R_dc_p - R_f_b - R_f_c, R_dc_p, 0, 0, 0, R_conv_b + R_f_b, R_conv_b, R_conv_b + R_f_b + R_f_c, 0, 1, 0, 1, -1,
			 0, 0, R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b, 1, 0, 1, -1, 0,
			 0, 0, R_conv_b, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_grid_b - R_src_b, -R_conv_b - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b + R_f_c, R_dc_n, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b, -R_conv_b - R_dc_n - R_dc_n_0, -R_conv_b - R_dc_n - R_dc_n_0 - R_f_b - R_f_c - R_grid_c - R_src_c, 1, 0, 0, -1, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_259(Model_3L::Components const& c) //  S_0_a S_0_b S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, -R_conv_a - R_f_a, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_dc_p - R_f_a - R_f_c, R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 1, 1, 0, -1,
			 0, 0, R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_264(Model_3L::Components const& c) //  S_n_a S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_a - R_conv_c - R_dc_n - R_dc_p - R_f_a - R_f_c, R_dc_n + R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 1, 1, 1, 0, -1,
			 0, 0, R_dc_n + R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, 0, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 1, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_266(Model_3L::Components const& c) //  S_0_b S_n_a S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_f_a - R_f_b, -R_conv_a - R_dc_n - R_f_a, R_dc_n, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 1, 0, 1, -1, 0,
			 0, -R_conv_a - R_dc_n - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_dc_p - R_f_a - R_f_c, R_dc_n + R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 1, 1, 1, 0, -1,
			 0, R_dc_n, R_dc_n + R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_dc_n, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_dc_n, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_272(Model_3L::Components const& c) //  S_n_b S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_b + L_conv_c, 0, 0, 0, 0, -L_conv_b, -L_conv_b, -L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_b - R_conv_c - R_dc_n - R_dc_p - R_f_b - R_f_c, R_dc_n + R_dc_p, 0, 0, 0, R_conv_b + R_f_b, R_conv_b, R_conv_b + R_f_b + R_f_c, 1, 1, 0, 1, -1,
			 0, 0, R_dc_n + R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b, 0, 0, 0, 0, -R_conv_b - R_dc_n_0 - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_f_b, 0, 0, 1, -1, 0,
			 0, 0, R_conv_b, 0, 0, 0, 0, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_grid_b - R_src_b, -R_conv_b - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b + R_f_c, 0, 0, 0, 0, -R_conv_b - R_dc_n_0 - R_f_b, -R_conv_b - R_dc_n_0, -R_conv_b - R_dc_n_0 - R_f_b - R_f_c - R_grid_c - R_src_c, 0, 0, 0, -1, 1,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_273(Model_3L::Components const& c) //  S_0_a S_n_b S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_f_a - R_f_b, -R_conv_a - R_f_a, -R_dc_n, 0, 0, 0, R_conv_a + R_dc_n, R_conv_a + R_dc_n + R_f_a + R_f_b, R_conv_a + R_dc_n + R_f_a, -1, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_dc_p - R_f_a - R_f_c, R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 1, 1, 0, -1,
			 0, -R_dc_n, R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_n, R_conv_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_f_a + R_f_b, R_conv_a + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_dc_n + R_f_a, R_conv_a + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_280(Model_3L::Components const& c) //  S_n_a S_n_b S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, -R_conv_a - R_f_a, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_dc_p - R_f_a - R_f_c, R_dc_n + R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 1, 1, 1, 0, -1,
			 0, 0, R_dc_n + R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_320(Model_3L::Components const& c) //  S_p_a S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_b, 0, 0, 0, -L_grid_b, 0, 0, -L_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, 0, R_conv_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, 0, R_conv_a + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, 0, R_conv_a + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, -R_grid_b - R_src_b, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 1, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_322(Model_3L::Components const& c) //  S_0_b S_p_a S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_f_a, -R_dc_p, 0, 0, 0, R_conv_a + R_dc_p, R_conv_a + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_p + R_f_a, 0, -1, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, -R_dc_p, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_p, R_conv_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_dc_p + R_f_a + R_f_b, R_conv_a + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_dc_p + R_f_a, R_conv_a + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_336(Model_3L::Components const& c) //  S_n_b S_p_a S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_f_a, -R_dc_n - R_dc_p, 0, 0, 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_b, R_conv_a + R_dc_n + R_dc_p + R_f_a, -1, -1, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, -R_dc_n - R_dc_p, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_dc_p, R_conv_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_dc_n + R_dc_p + R_f_a + R_f_b, R_conv_a + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_dc_n + R_dc_p + R_f_a, R_conv_a + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 1, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, -R_dc_n, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_384(Model_3L::Components const& c) //  S_p_b S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, 0, L_conv_b + L_conv_c, 0, 0, 0, 0, -L_conv_b, -L_conv_b, -L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b + L_grid_a + L_src_a, L_conv_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b + L_grid_b + L_src_b, L_conv_b, 0, 0, 0, 0, 0,
			 0, 0, -L_conv_b, 0, 0, 0, 0, L_conv_b, L_conv_b, L_conv_b + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 -L_conv_a, 0, 0, 0, -L_grid_a, 0, 0, -L_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, 0, -R_conv_b - R_conv_c - R_f_b - R_f_c, 0, 0, 0, 0, R_conv_b + R_f_b, R_conv_b, R_conv_b + R_f_b + R_f_c, 0, 0, 0, 1, -1,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_a - R_src_a, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b, 1, 1, 1, -1, 0,
			 0, 0, R_conv_b, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_b - R_src_b, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, 0, R_conv_b + R_f_b + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_b - R_dc_n - R_dc_n_0 - R_dc_p - R_f_b - R_f_c - R_grid_c - R_src_c, 1, 1, 0, -1, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, -R_grid_a - R_src_a, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 1, 0, 0,
			 1, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_385(Model_3L::Components const& c) //  S_0_a S_p_b S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_dc_p - R_f_a, R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 1, 1, -1, 0,
			 0, -R_conv_a - R_dc_p - R_f_a, -R_conv_a - R_conv_c - R_dc_p - R_f_a - R_f_c, R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 1, 1, 0, -1,
			 0, R_dc_p, R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n, R_dc_n, R_dc_n, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0, 1, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, 1, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, R_dc_n, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 0, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_392(Model_3L::Components const& c) //  S_n_a S_p_b S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_dc_n - R_dc_p - R_f_a - R_f_b, -R_conv_a - R_dc_n - R_dc_p - R_f_a, R_dc_n + R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 1, 1, 1, -1, 0,
			 0, -R_conv_a - R_dc_n - R_dc_p - R_f_a, -R_conv_a - R_conv_c - R_dc_n - R_dc_p - R_f_a - R_f_c, R_dc_n + R_dc_p, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 1, 1, 1, 0, -1,
			 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0 - R_grid_a - R_src_a, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n_0 - R_f_a, 0, 0, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 0, 0, -R_conv_a - R_dc_n_0, -R_conv_a - R_dc_n_0 - R_f_a, -R_conv_a - R_dc_n_0 - R_f_a - R_f_c - R_grid_c - R_src_c, 0, 0, -1, 0, 1,
			 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, R_dc_n, R_dc_n, -R_dc_n, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}


Model_3L::StateSpaceMatrices calculateStateSpace_448(Model_3L::Components const& c) //  S_p_a S_p_b S_p_c
{
	double L_conv_a = c.L_conv_a;
	double L_conv_b = c.L_conv_b;
	double L_conv_c = c.L_conv_c;
	double L_dc_src = c.L_dc_src;
	double L_grid_a = c.L_grid_a;
	double L_grid_b = c.L_grid_b;
	double L_grid_c = c.L_grid_c;
	double L_src_a = c.L_src_a;
	double L_src_b = c.L_src_b;
	double L_src_c = c.L_src_c;
	double C_dc_n = c.C_dc_n;
	double C_dc_p = c.C_dc_p;
	double C_f_a = c.C_f_a;
	double C_f_b = c.C_f_b;
	double C_f_c = c.C_f_c;
	double R_conv_a = c.R_conv_a;
	double R_conv_b = c.R_conv_b;
	double R_conv_c = c.R_conv_c;
	double R_dc_n = c.R_dc_n;
	double R_dc_p = c.R_dc_p;
	double R_dc_src = c.R_dc_src;
	double R_f_a = c.R_f_a;
	double R_f_b = c.R_f_b;
	double R_f_c = c.R_f_c;
	double R_grid_a = c.R_grid_a;
	double R_grid_b = c.R_grid_b;
	double R_grid_c = c.R_grid_c;
	double R_src_a = c.R_src_a;
	double R_src_b = c.R_src_b;
	double R_src_c = c.R_src_c;
	double R_dc_n_0 = c.R_dc_n_0;

    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> K1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_STATES> A1;
    Eigen::Matrix<double, Model_3L::NUM_STATES, Model_3L::NUM_INPUTS> B1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> K2;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_STATES> C1;
    Eigen::Matrix<double, Model_3L::NUM_OUTPUTS, Model_3L::NUM_INPUTS> D1;
        
    K1 <<
			 0, L_conv_a + L_conv_b, L_conv_a, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, L_conv_a, L_conv_a + L_conv_c, 0, 0, 0, 0, -L_conv_a, -L_conv_a, -L_conv_a, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a + L_grid_a + L_src_a, L_conv_a, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a + L_grid_b + L_src_b, L_conv_a, 0, 0, 0, 0, 0,
			 0, -L_conv_a, -L_conv_a, 0, 0, 0, 0, L_conv_a, L_conv_a, L_conv_a + L_grid_c + L_src_c, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_n, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_dc_p, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_a, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_b, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_f_c,
			 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0;
    K2 <<
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, L_dc_src, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    A1 <<
			 0, -R_conv_a - R_conv_b - R_f_a - R_f_b, -R_conv_a - R_f_a, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, 0, 0, 1, -1, 0,
			 0, -R_conv_a - R_f_a, -R_conv_a - R_conv_c - R_f_a - R_f_c, 0, 0, 0, 0, R_conv_a, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, 0, 0, 1, 0, -1,
			 0, 0, 0, -R_dc_n - R_dc_p - R_dc_src, 0, 0, 0, R_dc_n + R_dc_p, R_dc_n + R_dc_p, R_dc_n + R_dc_p, -1, -1, 0, 0, 0,
			 0, R_conv_a, R_conv_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_grid_a - R_src_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, 1, 1, 0, 0, 0,
			 0, R_conv_a + R_f_a + R_f_b, R_conv_a + R_f_a, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_b - R_grid_b - R_src_b, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, 1, 1, -1, 1, 0,
			 0, R_conv_a + R_f_a, R_conv_a + R_f_a + R_f_c, R_dc_n + R_dc_p, 0, 0, 0, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a, -R_conv_a - R_dc_n - R_dc_n_0 - R_dc_p - R_f_a - R_f_c - R_grid_c - R_src_c, 1, 1, -1, 0, 1,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0,
			 0, -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    B1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 -1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;
    C1 <<
			 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, -R_dc_n, 0, 0, 0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, R_dc_n + R_dc_n_0, -1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, R_dc_src, 0, 0, 0, R_dc_n_0, R_dc_n_0, R_dc_n_0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    D1 <<
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 1, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 1, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0,
			 0, 0, 0, 0;

    Model_3L::StateSpaceMatrices ss;
    ss.A = K1.partialPivLu().solve(A1);
    ss.B = K1.partialPivLu().solve(B1);
    ss.C = (C1 + K2 * ss.A);
    ss.D = (D1 + K2 * ss.B);
    return ss;
}

