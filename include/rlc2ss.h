// MIT License
//
// Copyright (c) 2024 vvainola
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include <string>
#include <vector>
#include <functional>
#include <bit>
#include <unordered_map>

#pragma warning(push, 0)
#include <Eigen/Core>
#pragma warning(pop)

#include "netlist/linear_expr.hpp"

namespace rlc2ss {

inline constexpr double MINIMUM_TIMESTEP = 1e-12;

std::string replace(const std::string& original, const std::string& search, const std::string& replacement);
std::vector<double> getCommaDelimitedValues(std::string const s);
double evaluateExpression(std::string expression);

struct ZeroCrossingEvent {
    double time;
    std::function<void(void)> event_callback;
    bool operator<(ZeroCrossingEvent const& other) const {
        return time < other.time;
    }
    bool operator>(ZeroCrossingEvent const& other) const {
        return time > other.time;
    }
};

double calcZeroCrossingTime(double y1, double y2);

/// State space matrices serialised as comma-delimited expression strings.
/// Used by the offline JSON-loading code paths (qucs/*_matrices.cpp non-dynamic).
struct StateSpaceMatrices {
    std::string K1;
    std::string K2;
    std::string A1;
    std::string B1;
    std::string C1;
    std::string D1;
};

/// State space matrices as typed symbolic-expression trees. Each entry of each
/// matrix is a `SymScalar`; downstream code calls `evaluate(SymbolicMatrix, vars)`
/// to substitute component values and produce an `Eigen::MatrixXd` directly,
/// without string substitution or parsing.
struct SymbolicStateSpace {
    SymbolicMatrix K1;
    SymbolicMatrix K2;
    SymbolicMatrix A1;
    SymbolicMatrix B1;
    SymbolicMatrix C1;
    SymbolicMatrix D1;
};

/// Form state-space matrices symbolically. The returned matrices contain
/// component names as atomic variables (e.g. "R1", "L1"). To get numeric
/// matrices, call `evaluate(...)` with a map from component name to value.
SymbolicStateSpace formStateSpaceMatrices(std::string const& netlist,
                                          uint64_t combination,
                                          bool verbose = false);

/// Evaluate a single symbolic scalar with component values substituted.
double evaluate(SymScalar const& s,
                std::unordered_map<std::string, double> const& values);

/// Evaluate a symbolic matrix with component values substituted. Uses a single
/// memoisation cache across all entries so that shared subexpressions (the AST
/// nodes referenced by `SymScalar::tree()`) are evaluated at most once.
Eigen::MatrixXd evaluate(SymbolicMatrix const& m,
                         std::unordered_map<std::string, double> const& values);

inline void hash_combine(uint64_t& seed, double v) {
    // Treat the double as its raw 64-bit representation
    // This avoids issues where the compiler might try to "help" with floating point logic
    uint64_t bits = std::bit_cast<uint64_t>(v);

    // Handle Negative Zero: if -0.0, treat it as 0.0 so they hash identically
    if (v == 0.0) {
        bits = std::bit_cast<uint64_t>(0.0);
    }

    seed ^= bits + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace rlc2ss
