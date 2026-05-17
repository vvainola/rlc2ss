// MIT License
//
// Copyright (c) 2026 vvainola
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

#include "netlist/sym_scalar.hpp"

#include <unordered_map>
#include <string>
#include <vector>
#include <format>

#pragma warning(push, 0)
#include "Eigen/Core"
#pragma warning(pop)

#define ALWAYS_UPDATE_STR 0 // For debugging to see the string representation at all times

namespace rlc2ss {

class LinearExpr {
  public:
    LinearExpr(std::string const& symbol_name) {
        terms[symbol_name] = SymScalar(1.0);
        updateStr();
    }
    LinearExpr(double c) {
        constant = SymScalar(c);
        updateStr();
    }
    LinearExpr(SymScalar c) {
        constant = std::move(c);
        updateStr();
    }
    LinearExpr() = default;

    std::string str() const {
        std::string result;
        for (auto const& [name, coeff] : terms) {
            if (coeff.isZero()) continue;
            if (!result.empty()) {
                result += "+";
            }
            if (coeff.isOne()) {
                result += name;
            } else if (coeff.isNumeric() && coeff.numeric() == -1.0) {
                // Remove trailing '+' if we just added one
                if (!result.empty() && result.back() == '+') {
                    result.pop_back();
                }
                result += std::format("-{}", name);
            } else {
                result += std::format("{}*{}", coeff.str(), name);
            }
        }
        if (!constant.isZero()) {
            if (!result.empty()) {
                result += "+";
            }
            result += constant.str();
        }
        return result.empty() ? "0" : result;
    }

    void updateStr() {
#if ALWAYS_UPDATE_STR
        m_str = str();
#endif
    }

    LinearExpr operator+(const LinearExpr& other) const {
        LinearExpr result = *this;
        for (auto const& [name, coeff] : other.terms) {
            result.terms[name] += coeff;
            if (result.terms[name].isZero()) {
                result.terms.erase(name);
            }
        }
        result.constant += other.constant;
        result.updateStr();
        return result;
    }

    LinearExpr& operator+=(const LinearExpr& other) {
        for (auto const& [name, coeff] : other.terms) {
            terms[name] += coeff;
            if (terms[name].isZero()) {
                terms.erase(name);
            }
        }
        constant += other.constant;
        updateStr();
        return *this;
    }

    LinearExpr operator-(const LinearExpr& other) const {
        LinearExpr result = *this;
        for (auto const& [name, coeff] : other.terms) {
            result.terms[name] -= coeff;
            if (result.terms[name].isZero()) {
                result.terms.erase(name);
            }
        }
        result.constant -= other.constant;
        result.updateStr();
        return result;
    }

    LinearExpr& operator-=(const LinearExpr& other) {
        for (auto const& [name, coeff] : other.terms) {
            terms[name] -= coeff;
            if (terms[name].isZero()) {
                terms.erase(name);
            }
        }
        constant -= other.constant;
        updateStr();
        return *this;
    }

    LinearExpr operator*(SymScalar scalar) const {
        if (scalar.isZero()) return LinearExpr(0.0);
        LinearExpr result = *this;
        for (auto it = result.terms.begin(); it != result.terms.end();) {
            it->second *= scalar;
            if (it->second.isZero()) {
                it = result.terms.erase(it);
            } else {
                ++it;
            }
        }
        result.constant *= scalar;
        result.updateStr();
        return result;
    }

    LinearExpr operator*(double scalar) const {
        return *this * SymScalar(scalar);
    }

    LinearExpr operator/(SymScalar scalar) const {
        assert(!scalar.isZero());
        if (scalar.isOne()) return *this;
        LinearExpr result = *this;
        for (auto it = result.terms.begin(); it != result.terms.end();) {
            it->second /= scalar;
            if (it->second.isZero()) {
                it = result.terms.erase(it);
            } else {
                ++it;
            }
        }
        result.constant /= scalar;
        result.updateStr();
        return result;
    }

    LinearExpr operator/(double scalar) const {
        return *this / SymScalar(scalar);
    }

    void replace(const std::string& symbol_name, const LinearExpr& new_expr) {
        if (terms.contains(symbol_name)) {
            SymScalar coeff = terms[symbol_name];
            terms.erase(symbol_name);
            // Add new_expr scaled by coeff
            for (auto const& [name, new_coeff] : new_expr.terms) {
                terms[name] += coeff * new_coeff;
                if (terms[name].isZero()) {
                    terms.erase(name);
                }
            }
            constant += coeff * new_expr.constant;
        }
        updateStr();
    }

  public:
#if ALWAYS_UPDATE_STR
    std::string m_str;
#endif
    std::unordered_map<std::string, SymScalar> terms; // symbol_name -> coefficient
    SymScalar constant;
};

inline LinearExpr operator*(SymScalar const& scalar, const LinearExpr& expr) {
    return expr * scalar;
}

inline LinearExpr operator*(double const scalar, const LinearExpr& expr) {
    return expr * scalar;
}

inline LinearExpr operator-(const LinearExpr& expr) {
    return expr * -1.0;
}

std::vector<LinearExpr> operator*(const Eigen::MatrixXd& mat, const std::vector<LinearExpr>& vec);

struct SymbolicMatrix {
    std::vector<std::vector<SymScalar>> data;
    int rows() const { return (int)data.size(); }
    int cols() const { return data.empty() ? 0 : (int)data[0].size(); }

    SymScalar& operator()(int r, int c) { return data[r][c]; }
    SymScalar const& operator()(int r, int c) const { return data[r][c]; }

    static SymbolicMatrix Zero(int rows, int cols) {
        SymbolicMatrix m;
        m.data.resize(rows, std::vector<SymScalar>(cols, SymScalar(0.0)));
        return m;
    }
};

struct SymbolicSystem {
    SymbolicMatrix A;
    std::vector<LinearExpr> b;
};

SymbolicSystem linearEqsToMatrix(const std::vector<LinearExpr>& eqns, const std::vector<std::string>& unknowns);
std::vector<LinearExpr> solveLinearSystem(SymbolicMatrix A, std::vector<LinearExpr> b, const std::vector<std::string>& unknowns);

} // namespace rlc2ss
