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

#include "netlist/sym_scalar.hpp"

#include <cassert>
#include <cmath>
#include <format>

namespace rlc2ss {

namespace {

ExprNodePtr makeConst(double v) {
    auto n = std::make_shared<ExprNode>();
    n->op = ExprNode::Op::Const;
    n->value = v;
    return n;
}

ExprNodePtr makeVar(std::string name) {
    auto n = std::make_shared<ExprNode>();
    n->op = ExprNode::Op::Var;
    n->name = std::move(name);
    return n;
}

ExprNodePtr makeBinary(ExprNode::Op op, ExprNodePtr lhs, ExprNodePtr rhs) {
    auto n = std::make_shared<ExprNode>();
    n->op = op;
    n->lhs = std::move(lhs);
    n->rhs = std::move(rhs);
    return n;
}

ExprNodePtr makeUnary(ExprNode::Op op, ExprNodePtr child) {
    auto n = std::make_shared<ExprNode>();
    n->op = op;
    n->lhs = std::move(child);
    return n;
}

// True iff the node renders as a single atom (no top-level operator),
// so it can safely appear without surrounding parens.
bool isAtom(ExprNode const& n) {
    return n.op == ExprNode::Op::Var
        || n.op == ExprNode::Op::Const
        || n.op == ExprNode::Op::Sqrt;
}

bool isAddOrSubTop(ExprNode const& n) {
    return n.op == ExprNode::Op::Add || n.op == ExprNode::Op::Sub;
}

bool structurallyEqual(ExprNode const& lhs, ExprNode const& rhs) {
    if (&lhs == &rhs) return true;
    if (lhs.op != rhs.op) return false;

    switch (lhs.op) {
        case ExprNode::Op::Var:
            return lhs.name == rhs.name;
        case ExprNode::Op::Const:
            return lhs.value == rhs.value;
        case ExprNode::Op::Neg:
        case ExprNode::Op::Sqrt:
            return lhs.lhs && rhs.lhs && structurallyEqual(*lhs.lhs, *rhs.lhs);
        case ExprNode::Op::Add:
        case ExprNode::Op::Sub:
        case ExprNode::Op::Mul:
        case ExprNode::Op::Div:
            return lhs.lhs && lhs.rhs && rhs.lhs && rhs.rhs
                && structurallyEqual(*lhs.lhs, *rhs.lhs)
                && structurallyEqual(*lhs.rhs, *rhs.rhs);
    }
    return false;
}

bool structurallyZero(ExprNode const& n) {
    switch (n.op) {
        case ExprNode::Op::Const:
            return n.value == 0.0;
        case ExprNode::Op::Neg:
        case ExprNode::Op::Sqrt:
            return n.lhs && structurallyZero(*n.lhs);
        case ExprNode::Op::Add:
            return n.lhs && n.rhs && structurallyZero(*n.lhs) && structurallyZero(*n.rhs);
        case ExprNode::Op::Sub:
            return n.lhs && n.rhs
                && (structurallyEqual(*n.lhs, *n.rhs)
                    || (structurallyZero(*n.lhs) && structurallyZero(*n.rhs)));
        case ExprNode::Op::Mul:
            return n.lhs && n.rhs && (structurallyZero(*n.lhs) || structurallyZero(*n.rhs));
        case ExprNode::Op::Div:
            return n.lhs && structurallyZero(*n.lhs);
        case ExprNode::Op::Var:
            return false;
    }
    return false;
}

// Append the textual representation of node into out with enough parentheses
// to preserve expression semantics for downstream string consumers.
void appendStr(std::string& out, ExprNode const& n) {
    switch (n.op) {
        case ExprNode::Op::Var:
            out += n.name;
            return;
        case ExprNode::Op::Const: {
            if (n.value == 0.0) {
                out += "0";
            } else if (n.value == 1.0) {
                out += "1";
            } else {
                out += std::format("{}", n.value);
            }
            return;
        }
        case ExprNode::Op::Add:
            appendStr(out, *n.lhs);
            out += '+';
            appendStr(out, *n.rhs);
            return;
        case ExprNode::Op::Sub:
            appendStr(out, *n.lhs);
            out += "-(";
            appendStr(out, *n.rhs);
            out += ')';
            return;
        case ExprNode::Op::Neg:
            if (isAtom(*n.lhs)) {
                out += '-';
                appendStr(out, *n.lhs);
            } else {
                out += "-(";
                appendStr(out, *n.lhs);
                out += ')';
            }
            return;
        case ExprNode::Op::Mul: {
            bool lhs_needs_parens = isAddOrSubTop(*n.lhs);
            bool rhs_needs_parens = isAddOrSubTop(*n.rhs);
            if (lhs_needs_parens) out += '(';
            appendStr(out, *n.lhs);
            if (lhs_needs_parens) out += ')';
            out += '*';
            if (rhs_needs_parens) out += '(';
            appendStr(out, *n.rhs);
            if (rhs_needs_parens) out += ')';
            return;
        }
        case ExprNode::Op::Div: {
            bool lhs_needs_parens = isAddOrSubTop(*n.lhs);
            if (lhs_needs_parens) out += '(';
            appendStr(out, *n.lhs);
            if (lhs_needs_parens) out += ')';
            out += '/';
            if (isAtom(*n.rhs)) {
                appendStr(out, *n.rhs);
            } else {
                out += '(';
                appendStr(out, *n.rhs);
                out += ')';
            }
            return;
        }
        case ExprNode::Op::Sqrt:
            out += "sqrt(";
            appendStr(out, *n.lhs);
            out += ')';
            return;
    }
}

} // namespace

SymScalar SymScalar::fromTree(ExprNodePtr tree) {
    SymScalar r;
    r.m_tree = std::move(tree);
    return r;
}

ExprNodePtr SymScalar::asTreeNode() const {
    if (m_tree) return m_tree;
    return makeConst(m_numeric);
}

SymScalar::SymScalar() = default;

SymScalar::SymScalar(double v)
    : m_numeric(v) {}

SymScalar::SymScalar(std::string name) {
    if (name == "0") {
        m_numeric = 0.0;
    } else if (name == "1") {
        m_numeric = 1.0;
    } else {
        m_numeric = 0.0;
        m_tree = makeVar(std::move(name));
    }
}

double SymScalar::numeric() const {
    assert(isNumeric());
    return m_numeric;
}

std::string const& SymScalar::str() const {
    if (m_cached_str) return *m_cached_str;
    std::string out;
    if (isNumeric()) {
        if (isZero()) {
            out = "0";
        } else if (isOne()) {
            out = "1";
        } else {
            out = std::format("{}", m_numeric);
        }
    } else {
        appendStr(out, *m_tree);
    }
    m_cached_str = std::move(out);
    return *m_cached_str;
}

bool SymScalar::isZero() const {
    if (isNumeric()) return m_numeric == 0.0;
    return m_tree && structurallyZero(*m_tree);
}

bool SymScalar::isOne() const {
    return isNumeric() && m_numeric == 1.0;
}

SymScalar SymScalar::operator+(SymScalar const& other) const {
    if (isZero()) return other;
    if (other.isZero()) return *this;
    if (isNumeric() && other.isNumeric()) return SymScalar(m_numeric + other.m_numeric);
    return fromTree(makeBinary(ExprNode::Op::Add, asTreeNode(), other.asTreeNode()));
}

SymScalar SymScalar::operator-(SymScalar const& other) const {
    if (other.isZero()) return *this;
    if (isZero()) return -other;
    if (isNumeric() && other.isNumeric()) return SymScalar(m_numeric - other.m_numeric);
    // x - x == 0, even when equal symbols were created as separate nodes.
    if (structurallyEqual(*asTreeNode(), *other.asTreeNode())) return SymScalar(0.0);
    return fromTree(makeBinary(ExprNode::Op::Sub, asTreeNode(), other.asTreeNode()));
}

SymScalar SymScalar::operator-() const {
    if (isZero()) return *this;
    if (isNumeric()) return SymScalar(-m_numeric);
    return fromTree(makeUnary(ExprNode::Op::Neg, asTreeNode()));
}

SymScalar SymScalar::operator*(SymScalar const& other) const {
    if (isZero() || other.isZero()) return SymScalar(0.0);
    if (isOne()) return other;
    if (other.isOne()) return *this;
    if (isNumeric() && other.isNumeric()) return SymScalar(m_numeric * other.m_numeric);
    // -1 * x == -x  (avoids producing "-1*..." which would surprise the parser)
    if (isNumeric() && m_numeric == -1.0) return -other;
    if (other.isNumeric() && other.m_numeric == -1.0) return -*this;
    return fromTree(makeBinary(ExprNode::Op::Mul, asTreeNode(), other.asTreeNode()));
}

SymScalar SymScalar::operator/(SymScalar const& other) const {
    assert(!other.isZero() && "Division by zero in SymScalar");
    if (isZero()) return SymScalar(0.0);
    if (other.isOne()) return *this;
    if (isNumeric() && other.isNumeric()) return SymScalar(m_numeric / other.m_numeric);
    // x / x == 1, even when equal symbols were created as separate nodes.
    if (structurallyEqual(*asTreeNode(), *other.asTreeNode())) return SymScalar(1.0);
    return fromTree(makeBinary(ExprNode::Op::Div, asTreeNode(), other.asTreeNode()));
}

SymScalar& SymScalar::operator+=(SymScalar const& other) {
    *this = *this + other;
    return *this;
}
SymScalar& SymScalar::operator-=(SymScalar const& other) {
    *this = *this - other;
    return *this;
}
SymScalar& SymScalar::operator*=(SymScalar const& other) {
    *this = *this * other;
    return *this;
}
SymScalar& SymScalar::operator/=(SymScalar const& other) {
    *this = *this / other;
    return *this;
}

bool SymScalar::operator==(double v) const {
    if (v == 0.0) return isZero();
    if (v == 1.0) return isOne();
    return isNumeric() && m_numeric == v;
}

bool SymScalar::operator!=(double v) const {
    return !(*this == v);
}

SymScalar SymScalar::sqrt(SymScalar const& x) {
    if (x.isZero()) return SymScalar(0.0);
    if (x.isOne()) return SymScalar(1.0);
    if (x.isNumeric()) return SymScalar(std::sqrt(x.m_numeric));
    return fromTree(makeUnary(ExprNode::Op::Sqrt, x.asTreeNode()));
}

SymScalar operator*(double lhs, SymScalar const& rhs) {
    return SymScalar(lhs) * rhs;
}

SymScalar operator/(double lhs, SymScalar const& rhs) {
    return SymScalar(lhs) / rhs;
}

} // namespace rlc2ss
