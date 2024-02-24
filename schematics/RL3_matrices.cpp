
#include "RL3_matrices.hpp"

#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4189) // local variable is initialized but not referenced
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4408) // anonymous struct did not declare any data members
#pragma warning(disable : 5054) // operator '&': deprecated between enumerations of different types

static std::unique_ptr<Model_RL3::StateSpaceMatrices> calcStateSpace(
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> const  &K1,
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> const  &A1,
    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_INPUTS> const  &B1,
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> const &K2,
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> const &C1,
    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_INPUTS> const &D1) {
    auto ss = std::make_unique<Model_RL3::StateSpaceMatrices>();
    ss->A   = K1.partialPivLu().solve(A1);
    ss->B   = K1.partialPivLu().solve(B1);
    ss->C   = (C1 + K2 * ss->A);
    ss->D   = (D1 + K2 * ss->B);
    return ss;
}

Model_RL3::Model_RL3(Components const& c)
    : components(c),
      m_components_DO_NOT_TOUCH(c) {
    m_ss = calculateStateSpace(components, switches);
    m_solver.updateJacobian(m_ss.A);
}

std::unique_ptr<Model_RL3::StateSpaceMatrices> calculateStateSpace_0(Model_RL3::Components const& c);

struct Model_RL3_Topology {
    Model_RL3::Components components;
    Model_RL3::Switches switches;
    std::unique_ptr<Model_RL3::StateSpaceMatrices> state_space;
};

Model_RL3::StateSpaceMatrices Model_RL3::calculateStateSpace(Model_RL3::Components const& components, Model_RL3::Switches switches)
{
    static std::vector<Model_RL3_Topology> state_space_cache;
    auto it = std::find_if(
        state_space_cache.begin(), state_space_cache.end(), [&](Model_RL3_Topology const& t) {
        return t.components == components && t.switches.all == switches.all;
    });
    if (it != state_space_cache.end()) {
        return *it->state_space;
    }
    auto state_space = std::make_unique<Model_RL3::StateSpaceMatrices>();

    switch (switches.all) {
		case 0: state_space = calculateStateSpace_0(components); break;
    default:
        assert(("Invalid switch combination", 0));
    }
    Model_RL3_Topology& topology = state_space_cache.emplace_back(Model_RL3_Topology{
        .components = components,
        .switches = switches,
        .state_space = std::move(state_space)});

    return *topology.state_space;
}

std::unique_ptr<Model_RL3::StateSpaceMatrices> calculateStateSpace_0(Model_RL3::Components const& c) // 
{
	double L_a = c.L_a;
	double L_b = c.L_b;
	double L_c = c.L_c;
	double R_a = c.R_a;
	double R_b = c.R_b;
	double R_c = c.R_c;
	double Kab = c.Kab;
	double Kbc = c.Kbc;
	double Kca = c.Kca;


    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> K1 {
		{ 0, -2*Kab*sqrt(L_a*L_b) + L_a + L_b, -Kab*sqrt(L_a*L_b) + Kbc*sqrt(L_b*L_c) - Kca*sqrt(L_a*L_c) + L_a },
		{ 0, -Kab*sqrt(L_a*L_b) + Kbc*sqrt(L_b*L_c) - Kca*sqrt(L_a*L_c) + L_a, -2*Kca*sqrt(L_a*L_c) + L_a + L_c },
		{ 1, 1, 1 } };

    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> K2 {
		{ 0, 0, 0 },
		{ 0, 0, 0 },
		{ 0, 0, 0} };

    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_STATES> A1 {
		{ 0, -R_a - R_b, -R_a },
		{ 0, -R_a, -R_a - R_c },
		{ 0, 0, 0 } };

    Eigen::Matrix<double, Model_RL3::NUM_STATES, Model_RL3::NUM_INPUTS> B1 {
		{ -1, 1, 0 },
		{ -1, 0, 1 },
		{ 0, 0, 0 } };

    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_STATES> C1 {
		{ 0, -1, -1 },
		{ 0, 1, 0 },
		{ 0, 0, 1 } };

    Eigen::Matrix<double, Model_RL3::NUM_OUTPUTS, Model_RL3::NUM_INPUTS> D1 {
		{ 0, 0, 0 },
		{ 0, 0, 0 },
		{ 0, 0, 0 } };

    return calcStateSpace(K1, A1, B1, K2, C1, D1);
}

