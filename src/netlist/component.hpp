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

#include "str_helpers.h"

#pragma warning(push, 0)
#include "symengine/expression.h"
#include "symengine/symbol.h"
#pragma warning(pop)

#include <vector>
#include <string>
#include <assert.h>
#include <format>

#define SYMBOLIC 0

namespace rlc2ss {

using namespace SymEngine;
inline constexpr std::string V_DUMMY = "V_switch";

class Component;
class Node;

class Node {
  public:
    Node(const std::string& name)
        : m_name(name) {}
    const std::string& name() const { return m_name; }
    void addConnection(Component* comp) { m_connections.push_back(comp); }
    const std::vector<Component*>& connections() const { return m_connections; }

  private:
    std::string m_name;
    std::vector<Component*> m_connections;
};

class Component {
  public:
    Component(const std::string& name,
              Node& pos_node,
              Node& neg_node,
              double value = 0)
        : m_name(name),
          m_pos_node(&pos_node),
          m_neg_node(&neg_node),
          m_value(value),
          m_voltage(std::format("V_{}", name)),
          m_current(std::format("I_{}", name)) {
        pos_node.addConnection(this);
        neg_node.addConnection(this);
        if (m_name[0] == 'V') {
            m_voltage = Expression(name);
        } else if (m_name[0] == 'I') {
            m_current = Expression(name);
        } else if (m_name[0] == 'R') {
            assert(m_value >= 0);
            m_voltage = SYMBOLIC ? m_name * m_current : m_value * m_current;
        }

        if (m_name[0] == 'C') {
            assert(m_value > 0);
            m_derivative = std::format("d{}", voltage().get_basic()->__str__());
        } else if (m_name[0] == 'L') {
            assert(m_value > 0);
            m_derivative = std::format("d{}", current().get_basic()->__str__());
            m_mutual_inductance_voltage = Expression("0");
        }

        if (name.starts_with(V_DUMMY)) {
            m_voltage = 0;
        }
    }

    const std::string& name() const { return m_name; }
    Node* posNode() const { return m_pos_node; }
    Node* negNode() const { return m_neg_node; }
    Expression derivative() const {
        assert(m_name[0] == 'L' || m_name[0] == 'C');
        return SYMBOLIC ? m_name * m_derivative : m_value * m_derivative;
    }

    Expression v_derivative() const {
        assert(m_name[0] == 'L');
        return SYMBOLIC ?
                   m_name * m_derivative + m_mutual_inductance_voltage :
                   m_value * m_derivative + m_mutual_inductance_voltage;
    }

    Expression i_derivative() const {
        assert(m_name[0] == 'C');
        return SYMBOLIC ? m_name * m_derivative : m_value * m_derivative;
    }

    Expression const& voltage() const { return m_voltage; }
    Expression const& current() const { return m_current; }
    void setVoltage(const Expression& expr) {
        m_voltage = SymEngine::expand(expr);
        if (m_name[0] == 'C') {
            m_derivative = str::replaceAll(m_voltage.get_basic()->__str__(), "V_", "dV_");
        }
    }
    void setCurrent(const Expression& expr) {
        m_current = SymEngine::expand(expr);
        if (m_name[0] == 'L') {
            m_derivative = str::replaceAll(m_current.get_basic()->__str__(), "I_", "dI_");
        } else if (m_name[0] == 'R') {
            m_voltage = SYMBOLIC ? m_name * m_current : m_value * m_current;
        }
    }

    void addMutualInductance(Expression const& expr) {
        m_mutual_inductance_voltage += expr;
    }

    void setSourceVoltageNodes(Node const* node_pos, Node const* node_neg) {
        m_pos_src = node_pos;
        m_neg_src = node_neg;
    }
    Node const* posSource() const { return m_pos_src; }
    Node const* negSource() const { return m_neg_src; }

    std::string value() const { return SYMBOLIC ? m_name : std::to_string(m_value); }

  private:
    std::string m_name;
    double m_value;
    Node* m_pos_node;
    Node* m_neg_node;
    Expression m_voltage;
    Expression m_current;

    Expression m_derivative = Expression(SymEngine::zero);
    Expression m_mutual_inductance_voltage = Expression(SymEngine::zero);
    Node const* m_pos_src = nullptr;
    Node const* m_neg_src = nullptr;
};
} // namespace rlc2ss
