
#include "RL3_matrices.hpp"
#include "rlc2ss.h"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<Model_RL3::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> const& K1,
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> const& A1,
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_INPUTS> const& B1,
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> const& K2,
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> const& C1,
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_INPUTS> const& D1) {
    auto ss = std::make_unique<Model_RL3::StateSpaceMatrices>();
    ss->A = K1.partialPivLu().solve(A1);
    ss->B = K1.partialPivLu().solve(B1);
    ss->C = (C1 + K2 * ss->A);
    ss->D = (D1 + K2 * ss->B);
    return ss;
}

Model_RL3::Model_RL3(Components const& c)
    : components(c),
      m_components_DO_NOT_TOUCH(c) {
    m_ss = calculateStateSpace(components, switches);
}

struct Model_RL3_Topology {
    Model_RL3::Components components;
    Model_RL3::Switches switches;
    std::unique_ptr<Model_RL3::StateSpaceMatrices> state_space;
};

Model_RL3::StateSpaceMatrices Model_RL3::calculateStateSpace(Model_RL3::Components const& components, Model_RL3::Switches switches) {
    static std::vector<Model_RL3_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_RL3_Topology const& t) {
            return t.components == components && t.switches.all == switches.all;
        });
    if (it != state_space_cache.end()) {
        return *it->state_space;
    }

    if (m_circuit_json.empty()) {
        m_circuit_json = nlohmann::json::parse(rlc2ss::loadTextResource(101));
    }
    assert(!m_circuit_json.empty());

    // Get the intermediate matrices as string for replacing symbolic components with their values
    std::string s = m_circuit_json[std::to_string(switches.all)].dump();
	s = rlc2ss::replace(s, "L_a", std::to_string(components.L_a));
	s = rlc2ss::replace(s, "L_b", std::to_string(components.L_b));
	s = rlc2ss::replace(s, "L_c", std::to_string(components.L_c));
	s = rlc2ss::replace(s, "R_a", std::to_string(components.R_a));
	s = rlc2ss::replace(s, "R_b", std::to_string(components.R_b));
	s = rlc2ss::replace(s, "R_c", std::to_string(components.R_c));
	s = rlc2ss::replace(s, "Kab", std::to_string(components.Kab));
	s = rlc2ss::replace(s, "Kbc", std::to_string(components.Kbc));
	s = rlc2ss::replace(s, "Kca", std::to_string(components.Kca));

    // Parse json for the intermediate matrices
    nlohmann::json j = nlohmann::json::parse(s);
    std::string K1_str = j["K1"];
    std::string K2_str = j["K2"];
    std::string A1_str = j["A1"];
    std::string B1_str = j["B1"];
    std::string C1_str = j["C1"];
    std::string D1_str = j["D1"];

    // Create eigen matrices
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES, Eigen::RowMajor> K1(rlc2ss::getCommaDelimitedValues(K1_str).data());
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES, Eigen::RowMajor> K2(rlc2ss::getCommaDelimitedValues(K2_str).data());
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES, Eigen::RowMajor> A1(rlc2ss::getCommaDelimitedValues(A1_str).data());
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_INPUTS, Eigen::RowMajor> B1(rlc2ss::getCommaDelimitedValues(B1_str).data());
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES, Eigen::RowMajor> C1(rlc2ss::getCommaDelimitedValues(C1_str).data());
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_INPUTS, Eigen::RowMajor> D1(rlc2ss::getCommaDelimitedValues(D1_str).data());

    Model_RL3_Topology& topology = state_space_cache.emplace_back(Model_RL3_Topology{
        .components = components,
        .switches = switches,
        .state_space = calcStateSpace(K1, A1, B1, K2, C1, D1)});

    return *topology.state_space;
}
