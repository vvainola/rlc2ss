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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace rlc2ss {

// Expression tree node. Owned via shared_ptr so common subexpressions
// (e.g. the "factor" reused across a row in Gaussian elimination) are
// referenced rather than copied.
struct ExprNode {
    enum class Op : std::uint8_t { Var, Const, Add, Sub, Neg, Mul, Div, Sqrt };
    Op op;
    double value = 0.0;                            // for Const
    std::string name;                              // for Var (and atom-sqrt)
    std::shared_ptr<ExprNode const> lhs;           // operand 1 (or sole operand for unary)
    std::shared_ptr<ExprNode const> rhs;           // operand 2 for binary
};

using ExprNodePtr = std::shared_ptr<ExprNode const>;

// Symbolic scalar represented as a numeric value (fast path) or a shared
// expression tree. Public API mirrors the previous string-based version so
// linear_expr.* and state_space.cpp need no changes.
class SymScalar {
  public:
    SymScalar();                                  // 0
    SymScalar(double v);                          // numeric constant
    explicit SymScalar(std::string name);         // named symbol (e.g. "R1")

    bool isZero() const { return m_is_zero; }
    bool isOne() const { return m_is_one; }
    bool isNumeric() const { return m_is_numeric; }
    double numeric() const;
    std::string const& str() const;

    SymScalar operator+(SymScalar const& other) const;
    SymScalar operator-(SymScalar const& other) const;
    SymScalar operator-() const;
    SymScalar operator*(SymScalar const& other) const;
    SymScalar operator/(SymScalar const& other) const;

    SymScalar& operator+=(SymScalar const& other);
    SymScalar& operator-=(SymScalar const& other);
    SymScalar& operator*=(SymScalar const& other);
    SymScalar& operator/=(SymScalar const& other);

    bool operator==(double v) const;
    bool operator!=(double v) const;

    // sqrt of a symbolic expression. Numeric values fold; symbolic expressions
    // become an opaque `sqrt(<inner>)` atom (algebraic simplification of nested
    // sqrt is out of scope).
    static SymScalar sqrt(SymScalar const& x);

    // Returns the underlying expression-tree node, or null when this scalar
    // is purely numeric (use `numeric()` in that case).
    ExprNodePtr tree() const { return m_tree; }

  private:
    static SymScalar fromTree(ExprNodePtr tree);
    ExprNodePtr asTreeNode() const;                // null if isZero

    bool m_is_zero = true;
    bool m_is_one = false;
    bool m_is_numeric = true;
    double m_numeric = 0.0;
    ExprNodePtr m_tree;                            // null if m_is_numeric
    mutable std::optional<std::string> m_cached_str;
};

SymScalar operator*(double lhs, SymScalar const& rhs);
SymScalar operator/(double lhs, SymScalar const& rhs);

} // namespace rlc2ss
