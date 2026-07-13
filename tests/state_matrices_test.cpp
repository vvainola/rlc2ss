#include <catch2/catch_test_macros.hpp>

#pragma warning(push, 0)
#include <Eigen/Core>
#include <Eigen/LU>
#pragma warning(pop)
#include <nlohmann/json.hpp>

#include "rlc2ss.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef RLC2SS_QUCS_DIR
#error "RLC2SS_QUCS_DIR must be defined to point at the qucs/ directory"
#endif

namespace {

std::string slurpFile(std::filesystem::path const& p) {
    std::ifstream f(p);
    REQUIRE(f.is_open());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Substitute component name -> value in a symbolic expression string and
// evaluate the resulting comma-delimited numeric expressions to a flat vector.
std::vector<double> evaluateSymbolicCsv(std::string s,
                                        std::unordered_map<std::string, double> const& values) {
    // Replace longer names first so e.g. "R_D_a_p" is substituted before "R_a"
    std::vector<std::string> keys;
    keys.reserve(values.size());
    for (auto const& [k, _] : values) {
        keys.push_back(k);
    }
    std::sort(keys.begin(), keys.end(), [](std::string const& a, std::string const& b) {
        return a.size() > b.size();
    });
    for (auto const& k : keys) {
        s = rlc2ss::replace(s, k, std::format("({})", values.at(k)));
    }
    return rlc2ss::getCommaDelimitedValues(s);
}

Eigen::MatrixXd toRowMajorMatrix(std::vector<double> const& flat, int rows, int cols) {
    REQUIRE(flat.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols));
    Eigen::MatrixXd m(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            m(i, j) = flat[static_cast<size_t>(i) * cols + j];
        }
    }
    return m;
}

struct Fixture {
    std::string cir_filename;
    std::string json_filename;
    int num_states;
    int num_inputs;
    int num_outputs;
    std::unordered_map<std::string, double> component_values;
};

// Solve the intermediate matrices into final state-space form:
//   A = K1^-1 A1, B = K1^-1 B1, C = C1 + K2 A, D = D1 + K2 B.
// This matches the production calcStateSpace helper (see qucs/*_matrices.cpp).
// Row permutations of K1/A1/B1 cancel out here, so this comparison is
// invariant to the cap-cutset-vs-inductor-loop ordering difference between
// the Python and C++ pipelines.
struct AbcdMatrices {
    Eigen::MatrixXd A, B, C, D;
};

AbcdMatrices solveStateSpace(Eigen::MatrixXd const& K1,
                              Eigen::MatrixXd const& K2,
                              Eigen::MatrixXd const& A1,
                              Eigen::MatrixXd const& B1,
                              Eigen::MatrixXd const& C1,
                              Eigen::MatrixXd const& D1) {
    AbcdMatrices m;
    auto lu = K1.partialPivLu();
    m.A = lu.solve(A1);
    m.B = lu.solve(B1);
    m.C = C1 + K2 * m.A;
    m.D = D1 + K2 * m.B;
    return m;
}

Eigen::MatrixXd symbolicCsvToMatrix(std::string const& csv,
                                    std::unordered_map<std::string, double> const& values,
                                    int rows,
                                    int cols) {
    return toRowMajorMatrix(evaluateSymbolicCsv(csv, values), rows, cols);
}

bool closeEnough(Eigen::MatrixXd const& a, Eigen::MatrixXd const& b) {
    return (a - b).norm() <= 1e-9 * (a.norm() + 1.0);
}

void dumpIfDiff(char const* name, Eigen::MatrixXd const& a, Eigen::MatrixXd const& b) {
    if (!closeEnough(a, b)) {
        std::ostringstream os;
        os << name << " python:\n" << a << "\n" << name << " c++:\n" << b << "\n";
        UNSCOPED_INFO(os.str());
    }
}

void compareCombination(Fixture const& fx, std::string const& combination_key, int combination) {
    CAPTURE(fx.cir_filename, combination_key);
    auto qucs_dir = std::filesystem::path(RLC2SS_QUCS_DIR);
    std::string netlist = slurpFile(qucs_dir / fx.cir_filename);
    nlohmann::json j = nlohmann::json::parse(slurpFile(qucs_dir / fx.json_filename));
    REQUIRE(j.contains(combination_key));
    auto const& j_combo = j[combination_key];

    auto py_matrix = [&](char const* key, int rows, int cols) {
        return symbolicCsvToMatrix(j_combo[key].template get<std::string>(),
                                   fx.component_values,
                                   rows,
                                   cols);
    };
    AbcdMatrices py = solveStateSpace(
        py_matrix("K1", fx.num_states,  fx.num_states),
        py_matrix("K2", fx.num_outputs, fx.num_states),
        py_matrix("A1", fx.num_states,  fx.num_states),
        py_matrix("B1", fx.num_states,  fx.num_inputs),
        py_matrix("C1", fx.num_outputs, fx.num_states),
        py_matrix("D1", fx.num_outputs, fx.num_inputs));

    rlc2ss::SymbolicStateSpace ss = rlc2ss::formStateSpaceMatrices(netlist, combination);
    AbcdMatrices cpp = solveStateSpace(
        rlc2ss::evaluate(ss.K1, fx.component_values),
        rlc2ss::evaluate(ss.K2, fx.component_values),
        rlc2ss::evaluate(ss.A1, fx.component_values),
        rlc2ss::evaluate(ss.B1, fx.component_values),
        rlc2ss::evaluate(ss.C1, fx.component_values),
        rlc2ss::evaluate(ss.D1, fx.component_values));

    dumpIfDiff("A", py.A, cpp.A);
    dumpIfDiff("B", py.B, cpp.B);
    dumpIfDiff("C", py.C, cpp.C);
    dumpIfDiff("D", py.D, cpp.D);
    CHECK(closeEnough(py.A, cpp.A));
    CHECK(closeEnough(py.B, cpp.B));
    CHECK(closeEnough(py.C, cpp.C));
    CHECK(closeEnough(py.D, cpp.D));
}

} // namespace

TEST_CASE("Mutual inductor: C++ matches Python JSON") {
    Fixture fx{
        .cir_filename = "mutual_inductor.cir",
        .json_filename = "mutual_inductor_matrices.json",
        .num_states = 4,
        .num_inputs = 4,
        .num_outputs = 10,
        .component_values = {
            {"Cf", 1e-4},
            {"FSRC1", 10.0},
            {"K12", 0.5},
            {"K21", 0.3},
            {"K31", 0.2},
            {"L1", 1.0},
            {"L2", 1.5},
            {"L3", 2.0},
            {"R1", 10.0},
            {"R2", 12.0},
            {"R3", 0.01},
            {"R4", 10.0},
        },
    };
    compareCombination(fx, "0", 0);
}

TEST_CASE("Mutual inductance contributes to inductor-voltage outputs") {
    std::string const netlist = R"(
V1 P1 0 DC 1
R1 P1 N1 1
L1 N1 0 1;VC;
V2 P2 0 DC 1
R2 P2 N2 1
L2 N2 0 4;VC;
K12 L1 L2 0.5
)";

    rlc2ss::SymbolicStateSpace ss = rlc2ss::formStateSpaceMatrices(netlist, 0);
    Eigen::MatrixXd K2 = rlc2ss::evaluate(ss.K2, {
        {"L1", 1.0},
        {"L2", 4.0},
        {"K12", 0.5},
        {"R1", 1.0},
        {"R2", 1.0},
    });

    // Outputs are sorted as I_L1, I_L2, N1, N2, V_L1, V_L2.  With
    // M = K12 * sqrt(L1 * L2) = 1, the two inductor-voltage rows must be
    // [L1, M] and [M, L2], not merely the diagonal self-inductance terms.
    REQUIRE(K2.rows() == 6);
    REQUIRE(K2.cols() == 2);
    CHECK(K2(4, 0) == 1.0);
    CHECK(K2(4, 1) == 1.0);
    CHECK(K2(5, 0) == 1.0);
    CHECK(K2(5, 1) == 4.0);
}

TEST_CASE("Converter: C++ matches Python JSON") {
    Fixture fx{
        .cir_filename = "converter.cir",
        .json_filename = "converter_matrices.json",
        .num_states = 11,
        .num_inputs = 10,
        .num_outputs = 28,
        .component_values = {
            {"C_a", 10e-3}, {"C_b", 10e-3}, {"C_c", 10e-3},
            {"C_n", 10e-3}, {"C_p", 10e-3},
            {"L_a", 1e-3}, {"L_b", 1e-3}, {"L_c", 1e-3},
            {"L_g_a", 1e-3}, {"L_g_b", 1e-3}, {"L_g_c", 1e-3},
            {"R_D_a_n", 0.5}, {"R_D_a_p", 0.5},
            {"R_D_b_n", 0.5}, {"R_D_b_p", 0.5},
            {"R_D_c_n", 0.5}, {"R_D_c_p", 0.5},
            {"R_a", 10e-3}, {"R_b", 10e-3}, {"R_c", 10e-3},
            {"R_dc", 1.0},
            {"R_g_a", 10e-3}, {"R_g_b", 10e-3}, {"R_g_c", 10e-3},
            {"R_n_p", 1e3}, {"R_n_s", 10e-3},
            {"R_p_p", 1e3}, {"R_p_s", 10e-3},
        },
    };

    SECTION("combination 0")    { compareCombination(fx, "0",    0); }
    SECTION("combination 1")    { compareCombination(fx, "1",    1); }
    SECTION("combination 7")    { compareCombination(fx, "7",    7); }
    SECTION("combination 15")   { compareCombination(fx, "15",   15); }
    SECTION("combination 31")   { compareCombination(fx, "31",   31); }
    SECTION("combination 47")   { compareCombination(fx, "47",   47); }
    SECTION("combination 55")   { compareCombination(fx, "55",   55); }
    SECTION("combination 62")   { compareCombination(fx, "62",   62); }
    SECTION("combination 63")   { compareCombination(fx, "63",   63); }
}
