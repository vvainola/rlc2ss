
#include "diode_bridge_3l_matrices.hpp"
#include "rlc2ss.h"
#include "nlohmann/json.hpp"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<Model_diode_bridge_3l::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_STATES, Model_diode_bridge_3l::NUM_STATES> const& K1,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_STATES, Model_diode_bridge_3l::NUM_STATES> const& A1,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_STATES, Model_diode_bridge_3l::NUM_INPUTS> const& B1,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_OUTPUTS, Model_diode_bridge_3l::NUM_STATES> const& K2,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_OUTPUTS, Model_diode_bridge_3l::NUM_STATES> const& C1,
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_OUTPUTS, Model_diode_bridge_3l::NUM_INPUTS> const& D1) {
    auto ss = std::make_unique<Model_diode_bridge_3l::StateSpaceMatrices>();
    ss->A = K1.partialPivLu().solve(A1);
    ss->B = K1.partialPivLu().solve(B1);
    ss->C = (C1 + K2 * ss->A);
    ss->D = (D1 + K2 * ss->B);
    return ss;
}

Model_diode_bridge_3l::Model_diode_bridge_3l(Components const& c)
    : components(c),
      m_components_DO_NOT_TOUCH(c) {
    m_ss = calculateStateSpace(components, switches);
}

struct Model_diode_bridge_3l_Topology {
    Model_diode_bridge_3l::Components components;
    Model_diode_bridge_3l::Switches switches;
    std::unique_ptr<Model_diode_bridge_3l::StateSpaceMatrices> state_space;
};

Model_diode_bridge_3l::StateSpaceMatrices Model_diode_bridge_3l::calculateStateSpace(Model_diode_bridge_3l::Components const& components, Model_diode_bridge_3l::Switches switches) {
    static std::vector<Model_diode_bridge_3l_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_diode_bridge_3l_Topology const& t) {
            return t.components == components && t.switches.all == switches.all;
        });
    if (it != state_space_cache.end()) {
        return *it->state_space;
    }

    if (m_circuit_json.empty()) {
        m_circuit_json = rlc2ss::loadTextResource(101);
    }
    assert(!m_circuit_json.empty());

    // Replace symbolic components with their values before parsing the json
    std::string s = m_circuit_json;
	s = rlc2ss::replace(s, " L_conv_a ", std::to_string(components.L_conv_a));
	s = rlc2ss::replace(s, " L_conv_b ", std::to_string(components.L_conv_b));
	s = rlc2ss::replace(s, " L_conv_c ", std::to_string(components.L_conv_c));
	s = rlc2ss::replace(s, " L_dc_n ", std::to_string(components.L_dc_n));
	s = rlc2ss::replace(s, " L_dc_p ", std::to_string(components.L_dc_p));
	s = rlc2ss::replace(s, " L_dc_src ", std::to_string(components.L_dc_src));
	s = rlc2ss::replace(s, " L_grid_a ", std::to_string(components.L_grid_a));
	s = rlc2ss::replace(s, " L_grid_b ", std::to_string(components.L_grid_b));
	s = rlc2ss::replace(s, " L_grid_c ", std::to_string(components.L_grid_c));
	s = rlc2ss::replace(s, " L_src_a ", std::to_string(components.L_src_a));
	s = rlc2ss::replace(s, " L_src_b ", std::to_string(components.L_src_b));
	s = rlc2ss::replace(s, " L_src_c ", std::to_string(components.L_src_c));
	s = rlc2ss::replace(s, " C_dc_n1 ", std::to_string(components.C_dc_n1));
	s = rlc2ss::replace(s, " C_dc_n2 ", std::to_string(components.C_dc_n2));
	s = rlc2ss::replace(s, " C_dc_p1 ", std::to_string(components.C_dc_p1));
	s = rlc2ss::replace(s, " C_dc_p2 ", std::to_string(components.C_dc_p2));
	s = rlc2ss::replace(s, " C_f_a ", std::to_string(components.C_f_a));
	s = rlc2ss::replace(s, " C_f_b ", std::to_string(components.C_f_b));
	s = rlc2ss::replace(s, " C_f_c ", std::to_string(components.C_f_c));
	s = rlc2ss::replace(s, " R_conv_a ", std::to_string(components.R_conv_a));
	s = rlc2ss::replace(s, " R_conv_b ", std::to_string(components.R_conv_b));
	s = rlc2ss::replace(s, " R_conv_c ", std::to_string(components.R_conv_c));
	s = rlc2ss::replace(s, " R_dc_pn1 ", std::to_string(components.R_dc_pn1));
	s = rlc2ss::replace(s, " R_dc_pn2 ", std::to_string(components.R_dc_pn2));
	s = rlc2ss::replace(s, " R_dc_pp1 ", std::to_string(components.R_dc_pp1));
	s = rlc2ss::replace(s, " R_dc_pp2 ", std::to_string(components.R_dc_pp2));
	s = rlc2ss::replace(s, " R_dc_sn1 ", std::to_string(components.R_dc_sn1));
	s = rlc2ss::replace(s, " R_dc_sn2 ", std::to_string(components.R_dc_sn2));
	s = rlc2ss::replace(s, " R_dc_sp1 ", std::to_string(components.R_dc_sp1));
	s = rlc2ss::replace(s, " R_dc_sp2 ", std::to_string(components.R_dc_sp2));
	s = rlc2ss::replace(s, " R_dc_src ", std::to_string(components.R_dc_src));
	s = rlc2ss::replace(s, " R_f_a ", std::to_string(components.R_f_a));
	s = rlc2ss::replace(s, " R_f_b ", std::to_string(components.R_f_b));
	s = rlc2ss::replace(s, " R_f_c ", std::to_string(components.R_f_c));
	s = rlc2ss::replace(s, " R_grid_a ", std::to_string(components.R_grid_a));
	s = rlc2ss::replace(s, " R_grid_b ", std::to_string(components.R_grid_b));
	s = rlc2ss::replace(s, " R_grid_c ", std::to_string(components.R_grid_c));
	s = rlc2ss::replace(s, " R_src_a ", std::to_string(components.R_src_a));
	s = rlc2ss::replace(s, " R_src_b ", std::to_string(components.R_src_b));
	s = rlc2ss::replace(s, " R_src_c ", std::to_string(components.R_src_c));

    // Parse json for the intermediate matrices
    nlohmann::json j = nlohmann::json::parse(s);
    std::string K1_str = j[std::to_string(switches.all)]["K1"];
    std::string K2_str = j[std::to_string(switches.all)]["K2"];
    std::string A1_str = j[std::to_string(switches.all)]["A1"];
    std::string B1_str = j[std::to_string(switches.all)]["B1"];
    std::string C1_str = j[std::to_string(switches.all)]["C1"];
    std::string D1_str = j[std::to_string(switches.all)]["D1"];

    // Create eigen matrices
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_STATES, Model_diode_bridge_3l::NUM_STATES, Eigen::RowMajor> K1(rlc2ss::getCommaDelimitedValues(K1_str).data());
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_OUTPUTS, Model_diode_bridge_3l::NUM_STATES, Eigen::RowMajor> K2(rlc2ss::getCommaDelimitedValues(K2_str).data());
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_STATES, Model_diode_bridge_3l::NUM_STATES, Eigen::RowMajor> A1(rlc2ss::getCommaDelimitedValues(A1_str).data());
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_STATES, Model_diode_bridge_3l::NUM_INPUTS, Eigen::RowMajor> B1(rlc2ss::getCommaDelimitedValues(B1_str).data());
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_OUTPUTS, Model_diode_bridge_3l::NUM_STATES, Eigen::RowMajor> C1(rlc2ss::getCommaDelimitedValues(C1_str).data());
    Eigen::Matrix<double, Model_diode_bridge_3l::NUM_OUTPUTS, Model_diode_bridge_3l::NUM_INPUTS, Eigen::RowMajor> D1(rlc2ss::getCommaDelimitedValues(D1_str).data());

    Model_diode_bridge_3l_Topology& topology = state_space_cache.emplace_back(Model_diode_bridge_3l_Topology{
        .components = components,
        .switches = switches,
        .state_space = calcStateSpace(K1, A1, B1, K2, C1, D1)});

    return *topology.state_space;
}
