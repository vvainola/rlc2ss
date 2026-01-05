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

#include <unordered_map>
#include <string>
#include <vector>
#include <format>

#pragma warning(push, 0)
#include "Eigen/Core"
#pragma warning(pop)

namespace rlc2ss {

class LinearExpr {
  public:
    LinearExpr(std::string const& symbol_name) {
        terms[symbol_name] = 1.0;
    }
    LinearExpr(double c) {
        constant = c;
    }
    LinearExpr() = default;

    std::string str() const {
        std::string result;
        for (auto const& [name, coeff] : terms) {
            if (!result.empty() && coeff >= 0) {
                result += "+";
            }
            if (coeff == 1.0) {
                result += name;
                continue;
            } else if (coeff == -1.0) {
                result += std::format("-{}", name);
                continue;
            } else {
                result += std::format("{}*{}", coeff, name);
            }
        }
        if (constant != 0.0) {
            if (!result.empty() && constant >= 0) {
                result += "+";
            }
            result += std::format("{}", constant);
        }
        return result.empty() ? "0" : result;
    }
    void updateStr() {
    }

    LinearExpr operator+(const LinearExpr& other) const {
        LinearExpr result = *this;
        for (auto const& [name, coeff] : other.terms) {
            result.terms[name] += coeff;
            if (result.terms[name] == 0.0) {
                result.terms.erase(name);
            }
        }
        result.constant += other.constant;
        return result;
    }

    LinearExpr& operator+=(const LinearExpr& other) {
        for (auto const& [name, coeff] : other.terms) {
            terms[name] += coeff;
            if (terms[name] == 0.0) {
                terms.erase(name);
            }
        }
        constant += other.constant;
        return *this;
    }

    LinearExpr operator-(const LinearExpr& other) const {
        LinearExpr result = *this;
        for (auto const& [name, coeff] : other.terms) {
            result.terms[name] -= coeff;
            if (result.terms[name] == 0.0) {
                result.terms.erase(name);
            }
        }
        result.constant -= other.constant;
        return result;
    }

    LinearExpr& operator-=(const LinearExpr& other) {
        for (auto const& [name, coeff] : other.terms) {
            terms[name] -= coeff;
            if (terms[name] == 0.0) {
                terms.erase(name);
            }
        }
        constant -= other.constant;
        return *this;
    }

    LinearExpr operator*(double scalar) const {
        LinearExpr result = *this;
        for (auto& [name, coeff] : result.terms) {
            coeff *= scalar;
            if (coeff == 0.0) {
                result.terms.erase(name);
            }
        }
        result.constant *= scalar;
        return result;
    }

    LinearExpr operator/(double scalar) const {
        LinearExpr result = *this;
        for (auto& [name, coeff] : result.terms) {
            coeff /= scalar;
            if (coeff == 0.0) {
                result.terms.erase(name);
            }
        }
        result.constant /= scalar;
        return result;
    }

    void replace(const std::string& symbol_name, const LinearExpr& new_expr) {
        if (terms.contains(symbol_name)) {
            double coeff = terms[symbol_name];
            terms.erase(symbol_name);
            // Add new_expr scaled by coeff
            for (auto const& [name, new_coeff] : new_expr.terms) {
                terms[name] += coeff * new_coeff;
                if (terms[name] == 0.0) {
                    terms.erase(name);
                }
            }
            constant += coeff * new_expr.constant;
        }
    }

  public:
    std::unordered_map<std::string, double> terms; // symbol_name -> coefficient
    double constant = 0.0;
};

inline LinearExpr operator*(double const scalar, const LinearExpr& expr) {
    return expr * scalar;
}

inline LinearExpr operator-(const LinearExpr& expr) {
    return expr * -1.0;
}

std::vector<LinearExpr> operator*(const Eigen::MatrixXd& mat, const std::vector<LinearExpr>& vec);

struct SymbolicSystem {
    Eigen::MatrixXd A;
    std::vector<LinearExpr> b;
};

SymbolicSystem linearEqsToMatrix(const std::vector<LinearExpr>& eqns, const std::vector<std::string>& unknowns);

} // namespace rlc2ss
