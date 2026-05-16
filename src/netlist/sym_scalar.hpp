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

#include <string>
#include <format>
#include <cmath>
#include <cassert>

namespace rlc2ss {

/// A symbolic scalar expression represented as a string.
/// Supports basic arithmetic (+, -, *, /) and tracks special cases (zero, one)
/// for optimization. The string form is compatible with evaluateExpression().
class SymScalar {
  public:
    /// Construct zero
    SymScalar()
        : m_expr("0"), m_is_zero(true), m_is_one(false), m_numeric(0.0), m_is_numeric(true) {}

    /// Construct from double constant
    SymScalar(double v)
        : m_is_zero(v == 0.0), m_is_one(v == 1.0), m_numeric(v), m_is_numeric(true) {
        if (m_is_zero) {
            m_expr = "0";
        } else if (m_is_one) {
            m_expr = "1";
        } else {
            m_expr = std::format("{}", v);
        }
    }

    /// Construct from a named symbol (e.g. "R1", "L1")
    explicit SymScalar(std::string name)
        : m_expr(std::move(name)), m_is_zero(false), m_is_one(false), m_numeric(0.0), m_is_numeric(false) {
        // Check if the string is actually a numeric value
        if (m_expr == "0") {
            m_is_zero = true;
            m_is_numeric = true;
            m_numeric = 0.0;
        } else if (m_expr == "1") {
            m_is_one = true;
            m_is_numeric = true;
            m_numeric = 1.0;
        }
    }

    bool isZero() const { return m_is_zero; }
    bool isOne() const { return m_is_one; }
    bool isNumeric() const { return m_is_numeric; }
    double numeric() const {
        assert(m_is_numeric);
        return m_numeric;
    }
    std::string const& str() const { return m_expr; }

    SymScalar operator+(SymScalar const& other) const {
        if (m_is_zero) return other;
        if (other.m_is_zero) return *this;
        if (m_is_numeric && other.m_is_numeric) return SymScalar(m_numeric + other.m_numeric);
        SymScalar result;
        result.m_is_zero = false;
        result.m_is_one = false;
        result.m_is_numeric = false;
        result.m_expr = std::format("{}+{}", m_expr, other.m_expr);
        return result;
    }

    SymScalar operator-(SymScalar const& other) const {
        if (other.m_is_zero) return *this;
        if (m_is_zero) return -other;
        if (m_is_numeric && other.m_is_numeric) return SymScalar(m_numeric - other.m_numeric);
        // x - x = 0 (detect identical expression strings)
        if (m_expr == other.m_expr) return SymScalar(0.0);
        SymScalar result;
        result.m_is_zero = false;
        result.m_is_one = false;
        result.m_is_numeric = false;
        result.m_expr = std::format("{}-({})", m_expr, other.m_expr);
        return result;
    }

    SymScalar operator-() const {
        if (m_is_zero) return *this;
        if (m_is_numeric) return SymScalar(-m_numeric);
        SymScalar result;
        result.m_is_zero = false;
        result.m_is_one = false;
        result.m_is_numeric = false;
        if (isAtom()) {
            result.m_expr = std::format("-{}", m_expr);
        } else {
            result.m_expr = std::format("-({})", m_expr);
        }
        return result;
    }

    SymScalar operator*(SymScalar const& other) const {
        if (m_is_zero || other.m_is_zero) return SymScalar(0.0);
        if (m_is_one) return other;
        if (other.m_is_one) return *this;
        if (m_is_numeric && other.m_is_numeric) return SymScalar(m_numeric * other.m_numeric);
        // -1 * x = -x
        if (m_is_numeric && m_numeric == -1.0) return -other;
        if (other.m_is_numeric && other.m_numeric == -1.0) return -*this;
        SymScalar result;
        result.m_is_zero = false;
        result.m_is_one = false;
        result.m_is_numeric = false;
        std::string lhs = needsParensForMul() ? std::format("({})", m_expr) : m_expr;
        std::string rhs = other.needsParensForMul() ? std::format("({})", other.m_expr) : other.m_expr;
        result.m_expr = std::format("{}*{}", lhs, rhs);
        return result;
    }

    SymScalar operator/(SymScalar const& other) const {
        assert(!other.m_is_zero && "Division by zero");
        if (m_is_zero) return SymScalar(0.0);
        if (other.m_is_one) return *this;
        if (m_is_numeric && other.m_is_numeric) return SymScalar(m_numeric / other.m_numeric);
        // x / x = 1 (detect identical expression strings)
        if (m_expr == other.m_expr) return SymScalar(1.0);
        SymScalar result;
        result.m_is_zero = false;
        result.m_is_one = false;
        result.m_is_numeric = false;
        std::string lhs = needsParensForMul() ? std::format("({})", m_expr) : m_expr;
        // Denominator always gets parens unless it's an atom
        std::string rhs = other.isAtom() ? other.m_expr : std::format("({})", other.m_expr);
        result.m_expr = std::format("{}/{}", lhs, rhs);
        return result;
    }

    SymScalar& operator+=(SymScalar const& other) {
        *this = *this + other;
        return *this;
    }

    SymScalar& operator-=(SymScalar const& other) {
        *this = *this - other;
        return *this;
    }

    SymScalar& operator*=(SymScalar const& other) {
        *this = *this * other;
        return *this;
    }

    SymScalar& operator/=(SymScalar const& other) {
        *this = *this / other;
        return *this;
    }

    bool operator==(double v) const {
        if (v == 0.0) return m_is_zero;
        if (v == 1.0) return m_is_one;
        return m_is_numeric && m_numeric == v;
    }

    bool operator!=(double v) const {
        return !(*this == v);
    }

    /// sqrt of a symbolic expression
    static SymScalar sqrt(SymScalar const& x) {
        if (x.m_is_zero) return SymScalar(0.0);
        if (x.m_is_one) return SymScalar(1.0);
        if (x.m_is_numeric) return SymScalar(std::sqrt(x.m_numeric));
        SymScalar result;
        result.m_is_zero = false;
        result.m_is_one = false;
        result.m_is_numeric = false;
        result.m_expr = std::format("sqrt({})", x.m_expr);
        return result;
    }

  private:
    /// Returns true if expression is a simple atom (no operators at top level)
    bool isAtom() const {
        if (m_is_numeric) return true;
        // A symbol name with no operators
        for (char c : m_expr) {
            if (c == '+' || c == '-' || c == '*' || c == '/') return false;
        }
        return true;
    }

    /// Does this expression need parentheses when used as an operand of multiplication?
    bool needsParensForMul() const {
        if (m_is_numeric) {
            return m_numeric < 0; // Negative numbers need parens: (-3)*x
        }
        // If the expression contains + or - at top level, needs parens
        // Simple heuristic: if it contains + or - (not inside parens), it needs wrapping
        int depth = 0;
        for (char c : m_expr) {
            if (c == '(') depth++;
            else if (c == ')') depth--;
            else if (depth == 0 && (c == '+' || (c == '-' && &c != &m_expr[0]))) return true;
        }
        return false;
    }

    std::string m_expr;
    bool m_is_zero;
    bool m_is_one;
    double m_numeric;
    bool m_is_numeric;
};

inline SymScalar operator*(double lhs, SymScalar const& rhs) {
    return SymScalar(lhs) * rhs;
}

inline SymScalar operator/(double lhs, SymScalar const& rhs) {
    return SymScalar(lhs) / rhs;
}

} // namespace rlc2ss
