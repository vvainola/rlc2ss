// MIT License
//
// Copyright (c) 2025 vvainola
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
#include "rlc2ss.h"
#include "str_helpers.h"
#include "netlist.hpp"
#include "component.hpp"
#include "graph.hpp"

#pragma warning(push, 0)
#include "symengine/expression.h"
#include "symengine/matrix.h"
#include "symengine/symbol.h"
#include "symengine/real_double.h"
#include "symengine/solve.h"
#include "Eigen/Core"
#pragma warning(pop)

namespace rlc2ss {

using namespace SymEngine;

template <typename T>
inline bool contains(std::vector<T>& v, const T& item_to_search) {
    for (auto const& item : v) {
        if (item == item_to_search) {
            return true;
        }
    }
    return false;
}

template <typename T>
inline void appendVector(std::vector<T>& v, std::vector<T> const& append) {
    v.insert(v.end(), append.begin(), append.end());
}

DenseMatrix zeroMatrix(int rows, int cols) {
    DenseMatrix matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; ++j) {
            matrix.set(i, j, SymEngine::zero);
        }
    }
    return matrix;
}

static inline void replace(DenseMatrix& matrix, Expression const& old_eq, Expression const& new_eq) {
    for (unsigned int i = 0; i < matrix.nrows(); ++i) {
        for (unsigned int j = 0; j < matrix.ncols(); ++j) {
            Expression eq = matrix.get(i, j);
            eq = eq.subs({{old_eq, new_eq}});
            matrix.set(i, j, eq);
        }
    }
}

static inline void replace(SymEngine::vec_basic& vec, Expression const& old_eq, Expression const& new_eq) {
    for (auto& elem : vec) {
        elem = Expression(elem).subs({{old_eq, new_eq}});
    }
}

void print(std::vector<Expression> const& exprs) {
    for (auto& expr : exprs) {
        std::cout << expr.get_basic()->__str__() << std::endl;
    }
}

void print(std::vector<RCP<const Basic>> const& exprs) {
    for (auto& expr : exprs) {
        std::cout << expr->__str__() << std::endl;
    }
}

void print(std::vector<RCP<const Symbol>> const& exprs) {
    for (auto& expr : exprs) {
        std::cout << expr->__str__() << std::endl;
    }
}

void print(DenseMatrix const& exprs) {
    for (unsigned int i = 0; i < exprs.nrows(); ++i) {
        std::cout << exprs.get(i, 0)->__str__() << std::endl;
    }
}

std::vector<RCP<const Basic>> matrixToVector(DenseMatrix const& matrix) {
    assert(matrix.ncols() == 1);
    std::vector<RCP<const Basic>> vec;
    for (unsigned int i = 0; i < matrix.nrows(); ++i) {
        vec.push_back(matrix.get(i, 0));
    }
    return vec;
}

std::string matrixToStr(DenseMatrix const& matrix) {
    std::string result;
    for (unsigned int i = 0; i < matrix.nrows(); ++i) {
        for (unsigned int j = 0; j < matrix.ncols(); ++j) {
            result += matrix.get(i, j)->__str__();
            if (i != matrix.nrows() - 1 || j != matrix.ncols() - 1) {
                result += ", ";
            }
        }
    }
    return result;
}

Eigen::MatrixXd denseMatrixToEigenMatrix(DenseMatrix const& matrix) {
    Eigen::MatrixXd eigen_matrix(matrix.nrows(), matrix.ncols());
    for (unsigned int i = 0; i < matrix.nrows(); ++i) {
        for (unsigned int j = 0; j < matrix.ncols(); ++j) {
            RCP<const Basic> val = matrix.get(i, j);
            if (is_a<RealDouble>(*val)) {
                eigen_matrix(i, j) = down_cast<const RealDouble&>(*val).as_double();
            } else if (is_a<Integer>(*val)) {
                eigen_matrix(i, j) = static_cast<double>(down_cast<const Integer&>(*val).as_int());
            } else {
                throw std::runtime_error("Non-numeric value in denseMatrixToEigenMatrix: " + val->__str__());
            }
        }
    }
    return eigen_matrix;
}

DenseMatrix getRows(DenseMatrix const& matrix, std::vector<int> const& rows) {
    DenseMatrix result((int)rows.size(), matrix.ncols());
    for (unsigned int i = 0; i < rows.size(); ++i) {
        for (unsigned int j = 0; j < matrix.ncols(); ++j) {
            result.set(i, j, matrix.get(rows[i], j));
        }
    }
    return result;
}

std::vector<SymEngine::RCP<const SymEngine::Basic>> matrixVecMulSubset(DenseMatrix const& matrix,
                                                                       DenseMatrix const& vec,
                                                                       std::vector<int> const& rows) {
    SymEngine::DenseMatrix out_matrix = zeroMatrix(vec.nrows(), 1);
    DenseMatrix submatrix = getRows(matrix, rows);
    submatrix.mul_matrix(vec, out_matrix);
    std::vector<SymEngine::RCP<const SymEngine::Basic>> out_vec;
    for (int row = 0; row < rows.size(); ++row) {
        out_vec.push_back(out_matrix.get(row, 0));
    }
    return out_vec;
}

Expression nodeVoltage(Node const* node, Graph const& graph) {
    // Traverse from ground to node
    Expression node_voltage("0");
    if (node->name() == "0") {
        return node_voltage;
    }
    Node* ground = graph.getNode("0");
    Node* target = graph.getNode(node->name());
    assert(ground != nullptr);
    assert(target != nullptr);
    std::vector<Node*> path = graph.dijkstra(ground, target);
    for (int j = 1; j < path.size(); ++j) {
        Node* pos_node = path[j - 1];
        Node* neg_node = path[j];
        Component* comp = graph.getComponent(pos_node, neg_node);
        assert(comp != nullptr);
        if (comp->posNode() == pos_node) {
            node_voltage -= comp->voltage();
        } else {
            node_voltage += comp->voltage();
        }
    }
    return node_voltage;
}

// Solves Ax = b using Gaussian elimination
vec_basic solve_linear_system(DenseMatrix A, DenseMatrix b, const vec_sym& unknowns) {
    int n = unknowns.size();

    // Forward Elimination
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int pivot = i;
        while (pivot < n && is_a<Integer>(*(A.get(pivot, i))) && down_cast<const Integer&>(*(A.get(pivot, i))).is_zero())
            pivot++;

        if (pivot == n)
            continue;

        // Swap rows in A and b
        for (int k = 0; k < n; ++k) {
            auto temp = A.get(i, k);
            A.set(i, k, A.get(pivot, k));
            A.set(pivot, k, temp);
        }
        auto temp_b = b.get(i, 0);
        b.set(i, 0, b.get(pivot, 0));
        b.set(pivot, 0, temp_b);

        // Eliminate
        for (int j = i + 1; j < n; ++j) {
            auto factor = (div(A.get(j, i), A.get(i, i)));
            for (int k = i; k < n; ++k) {
                A.set(j, k, (sub(A.get(j, k), mul(factor, A.get(i, k)))));
            }
            b.set(j, 0, (sub(b.get(j, 0), mul(factor, b.get(i, 0)))));
        }
    }

    // Back Substitution
    vec_basic x(n);
    for (int i = n - 1; i >= 0; --i) {
        auto sum = b.get(i, 0);
        for (int j = i + 1; j < n; ++j) {
            sum = (sub(sum, mul(A.get(i, j), x[j])));
        }
        x[i] = expand(div(sum, A.get(i, i)));
    }
    return x;
}

// Solves Ax = b using Gaussian elimination (fast numeric version)
vec_basic solve_linear_system_fast(DenseMatrix A_sym, DenseMatrix b, const vec_sym& unknowns) {
    if (SYMBOLIC) {
        return solve_linear_system(A_sym, b, unknowns);
    }

    int n = unknowns.size();

    Eigen::MatrixXd A = denseMatrixToEigenMatrix(A_sym);

    // Forward Elimination
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int pivot = i;
        while (pivot < n && A(pivot, i) == 0)
            pivot++;

        if (pivot == n)
            continue;

        // Swap rows in A and b
        for (int k = 0; k < n; ++k) {
            auto temp = A(i, k);
            A(i, k) = A(pivot, k);
            A(pivot, k) = temp;
        }
        auto temp_b = b.get(i, 0);
        b.set(i, 0, b.get(pivot, 0));
        b.set(pivot, 0, temp_b);

        // Eliminate
        for (int j = i + 1; j < n; ++j) {
            auto factor = A(j, i) / A(i, i);
            for (int k = i; k < n; ++k) {
                A(j, k) = A(j, k) - factor * A(i, k);
            }
            b.set(j, 0, (sub(b.get(j, 0), mul(Expression(factor), b.get(i, 0)))));
        }
    }

    // Back Substitution
    vec_basic x(n);
    for (int i = n - 1; i >= 0; --i) {
        auto sum = b.get(i, 0);
        for (int j = i + 1; j < n; ++j) {
            sum = (sub(sum, mul(Expression(A(i, j)), x[j])));
        }
        x[i] = expand(div(sum, Expression(A(i, i))));
    }
    return x;
}

StateSpaceMatrices formStateSpaceMatrices(std::string const& netlist_str,
                                          int combination,
                                          std::unordered_map<std::string, double> const& component_values,
                                          bool verbose) {
    Netlist netlist = parseNetlist(netlist_str, combination, component_values);

    Graph full_graph;
    for (auto& comp : netlist.components) {
        full_graph.addComponent(comp.get());
    }

    // Add voltage source with zero voltage between ground and each node that is not reachable
    Node* ground = full_graph.getNode("0");
    for (std::unique_ptr<Node>& node : netlist.nodes) {
        bool has_path = full_graph.hasPath(ground, node.get());
        if (!has_path) {
            std::string new_name = V_DUMMY + node->name();
            std::unique_ptr<Component>& component = netlist.components.emplace_back(std::make_unique<Component>(new_name, *node, *ground, 0));
            netlist.voltage_sources.push_back(component.get());
            node->addConnection(component.get());
            ground->addConnection(component.get());
            full_graph.addComponent(component.get());
        }
    }

    /* Select nodes belonging to the "proper tree"
     * All branches corresponding to a voltage source must be selected.
     * The maximum possible number of branches corresponding to a capacitor should be selected.
     * Recall that the definition of a tree prohibits graph loops and as such, it may not be possible to include every capacitor of the network in the proper tree.
     * The maximum possible number of branches corresponding to a resistor should be selected such that the definition of tree is not violated.
     * The necessary number of branches corresponding to inductors and current sources required to complete the graph tree should be selected. */
    std::vector<Component*> twigs;
    std::vector<Component*> links;

    std::vector<Component*> voltage_source_like;
    voltage_source_like.insert(voltage_source_like.end(), netlist.voltage_sources.begin(), netlist.voltage_sources.end());
    voltage_source_like.insert(voltage_source_like.end(), netlist.vv_sources.begin(), netlist.vv_sources.end());
    voltage_source_like.insert(voltage_source_like.end(), netlist.iv_sources.begin(), netlist.iv_sources.end());
    // Add all voltage sources
    Graph proper_tree;
    for (Component* src : voltage_source_like) {
        twigs.push_back(src);
        proper_tree.addComponent(src);
    }
    std::vector<Component*> remaining_components;
    appendVector(remaining_components, netlist.capacitors);
    appendVector(remaining_components, netlist.resistors);
    appendVector(remaining_components, netlist.inductors);
    appendVector(remaining_components, netlist.current_sources);
    appendVector(remaining_components, netlist.ii_sources);
    appendVector(remaining_components, netlist.vi_sources);
    for (auto& comp : remaining_components) {
        bool has_path = proper_tree.hasPath(comp->posNode(), comp->negNode());
        if (!has_path) {
            twigs.push_back(comp);
            proper_tree.addComponent(comp);
        } else {
            links.push_back(comp);
        }
    }

    // State variables are capacitors that are in twigs and inductors that are in branches
    std::vector<SymEngine::RCP<const Symbol>> states;
    std::vector<Component*> dependent_capacitors;
    std::vector<Component*> dependent_inductors;
    std::unordered_map<std::string, SymEngine::Expression> output_currents;
    std::unordered_map<std::string, SymEngine::Expression> output_voltages;
    for (Component* cap : netlist.capacitors) {
        output_voltages[cap->voltage().get_basic()->__str__()] = cap->voltage();
        output_currents[cap->current().get_basic()->__str__()] = cap->current();
        if (contains(twigs, cap)) {
            states.push_back(SymEngine::make_rcp<const Symbol>(cap->voltage().get_basic()->__str__()));
        } else {
            dependent_capacitors.push_back(cap);
        }
    }
    for (Component* ind : netlist.inductors) {
        output_voltages[ind->voltage().get_basic()->__str__()] = ind->voltage();
        output_currents[ind->current().get_basic()->__str__()] = ind->current();
        if (!contains(twigs, ind)) {
            states.push_back(SymEngine::make_rcp<const Symbol>(ind->current().get_basic()->__str__()));
        } else {
            dependent_inductors.push_back(ind);
        }
    }

    // Create cutset matrix
    SymEngine::DenseMatrix cutset_matrix = zeroMatrix((int)twigs.size(), (int)netlist.components.size());
    std::vector<int> capacitor_cutset_rows;
    std::vector<int> dep_inductor_cutset_rows;
    std::vector<int> passive_cutset_rows;
    for (int i = 0; i < (int)twigs.size(); ++i) {
        // Remove twig
        Component* twig = twigs[i];
        proper_tree.removeComponent(twig);
        for (Component* link : links) {
            bool has_path = proper_tree.hasPath(link->posNode(), link->negNode());
            if (!has_path) {
                has_path = proper_tree.hasPath(twig->posNode(), link->posNode());
                if (has_path) {
                    cutset_matrix.set(i, netlist.getComponentIndex(link->name()), SymEngine::Expression("1"));
                } else {
                    cutset_matrix.set(i, netlist.getComponentIndex(link->name()), SymEngine::Expression("-1"));
                }
            }
        }
        cutset_matrix.set(i, netlist.getComponentIndex(twig->name()), SymEngine::Expression("1"));

        // Add twig back
        proper_tree.addComponent(twig);

        if (twig->name()[0] == 'C') {
            capacitor_cutset_rows.push_back(i);
        } else if (twig->name()[0] == 'L') {
            dep_inductor_cutset_rows.push_back(i);
        } else if (twig->name()[0] == 'I') {
            std::cout << std::format("Bad cutset: current source {} in twig!", twig->name()) << std::endl;
        } else {
            passive_cutset_rows.push_back(i);
        }
    }

    // Create loop matrix
    std::vector<int> inductor_loop_rows;
    std::vector<int> dep_capacitor_loop_rows;
    std::vector<int> passive_loop_rows;
    SymEngine::DenseMatrix loop_matrix = zeroMatrix((int)links.size(), (int)netlist.components.size());
    for (int i = 0; i < links.size(); ++i) {
        Component* link = links[i];
        std::vector<Node*> path = proper_tree.dijkstra(link->posNode(), link->negNode());
        for (int j = 1; j < path.size(); ++j) {
            Node* pos_node = path[j - 1];
            Node* neg_node = path[j];
            Component* comp = proper_tree.getComponent(pos_node, neg_node);
            if (comp->posNode()->name() == pos_node->name()) {
                loop_matrix.set(i, netlist.getComponentIndex(comp->name()), SymEngine::Expression("-1"));
            } else {
                loop_matrix.set(i, netlist.getComponentIndex(comp->name()), SymEngine::Expression("1"));
            }
        }
        Node* pos_node = path[path.size() - 1];
        if (link->posNode()->name() == pos_node->name()) {
            loop_matrix.set(i, netlist.getComponentIndex(link->name()), SymEngine::Expression("-1"));
        } else {
            loop_matrix.set(i, netlist.getComponentIndex(link->name()), SymEngine::Expression("1"));
        }

        if (link->name()[0] == 'C') {
            dep_capacitor_loop_rows.push_back(i);
        } else if (link->name()[0] == 'L') {
            inductor_loop_rows.push_back(i);
        } else {
            passive_loop_rows.push_back(i);
        }
    }

    // Replace voltage controlled sources with voltages over other components
    std::vector<Component*> voltage_controlled_sources;
    appendVector(voltage_controlled_sources, netlist.vv_sources);
    appendVector(voltage_controlled_sources, netlist.vi_sources);
    for (auto& src : voltage_controlled_sources) {
        full_graph.removeComponent(src);
        Expression pos_node_voltage = nodeVoltage(src->posSource(), full_graph);
        Expression neg_node_voltage = nodeVoltage(src->negSource(), full_graph);
        std::string gain = SYMBOLIC ? src->name() : src->value();
        if (src->name()[0] == 'E') {
            src->setVoltage(Expression(gain) * (pos_node_voltage - neg_node_voltage));
        } else if (src->name()[0] == 'G') {
            src->setCurrent(Expression(gain) * (pos_node_voltage - neg_node_voltage));
        }
        full_graph.addComponent(src);
    }

    // Form voltage and current vectors
    SymEngine::DenseMatrix i_vec = zeroMatrix((int)netlist.components.size(), 1);
    SymEngine::DenseMatrix u_vec = zeroMatrix((int)netlist.components.size(), 1);
    for (int i = 0; i < netlist.components.size(); ++i) {
        i_vec.set(i, 0, netlist.components[i]->current());
        u_vec.set(i, 0, netlist.components[i]->voltage());
    }

    // Collect equations corresponding to dependent components
    std::vector<SymEngine::RCP<const SymEngine::Basic>> dep_eqs;
    appendVector(dep_eqs, matrixVecMulSubset(cutset_matrix, i_vec, dep_inductor_cutset_rows));
    appendVector(dep_eqs, matrixVecMulSubset(loop_matrix, u_vec, dep_capacitor_loop_rows));
    if (verbose) {
        std::cout << "Dependent equations:" << std::endl;
        print(dep_eqs);
    }

    // Replace currents in inductors which are not state variables with state currents
    // Replace voltages in capacitors which are not state variables with state voltages
    std::vector<SymEngine::RCP<const Symbol>> dep_unknowns;
    for (Component* ind : dependent_inductors) {
        dep_unknowns.push_back(SymEngine::make_rcp<Symbol>(ind->current().get_basic()->__str__()));
    }
    for (Component* cap : dependent_capacitors) {
        dep_unknowns.push_back(SymEngine::make_rcp<Symbol>(cap->voltage().get_basic()->__str__()));
    }
    if (dep_unknowns.size() > 0) {
        auto [A, b] = SymEngine::linear_eqns_to_matrix(dep_eqs, dep_unknowns);
        std::vector<SymEngine::RCP<const SymEngine::Basic>> dep_solved = solve_linear_system_fast(A, b, dep_unknowns);
        assert(dep_solved.size() == dep_unknowns.size());
        for (unsigned int i = 0; i < dep_unknowns.size(); ++i) {
            std::string dep_unknown = dep_unknowns[i]->__str__();
            Component* comp = netlist.getComponent(dep_unknown.substr(2));
            if (dep_unknown[0] == 'I') {
                comp->setCurrent(Expression(dep_solved[i]));
                output_currents[dep_unknown] = dep_solved[i];
            } else if (dep_unknown[0] == 'V') {
                comp->setVoltage(Expression(dep_solved[i]));
                output_voltages[dep_unknown] = dep_solved[i];
            }
            replace(i_vec, Expression(dep_unknown), Expression(dep_solved[i]));
            replace(u_vec, Expression(dep_unknown), Expression(dep_solved[i]));
        }
    }

    /* Replace the passive components currents and voltages with state variables. */
    std::vector<SymEngine::RCP<const Symbol>> passive_unknowns;
    for (auto& current : netlist.current_sources) {
        passive_unknowns.push_back(SymEngine::make_rcp<Symbol>(current->voltage().get_basic()->__str__()));
    }
    for (auto& current : netlist.ii_sources) {
        passive_unknowns.push_back(SymEngine::make_rcp<Symbol>(current->voltage().get_basic()->__str__()));
    }
    for (auto& current : netlist.vi_sources) {
        passive_unknowns.push_back(SymEngine::make_rcp<Symbol>(current->voltage().get_basic()->__str__()));
    }
    for (auto& voltage : netlist.voltage_sources) {
        passive_unknowns.push_back(SymEngine::make_rcp<Symbol>(voltage->current().get_basic()->__str__()));
    }
    for (auto& voltage : netlist.vv_sources) {
        passive_unknowns.push_back(SymEngine::make_rcp<Symbol>(voltage->current().get_basic()->__str__()));
    }
    for (auto& voltage : netlist.iv_sources) {
        passive_unknowns.push_back(SymEngine::make_rcp<Symbol>(voltage->current().get_basic()->__str__()));
    }
    for (auto& resistor : netlist.resistors) {
        passive_unknowns.push_back(SymEngine::make_rcp<Symbol>(resistor->current().get_basic()->__str__()));
    }

    // Collect equations corresponding to passive components
    std::vector<SymEngine::RCP<const SymEngine::Basic>> passive_eqs;
    appendVector(passive_eqs, matrixVecMulSubset(cutset_matrix, i_vec, passive_cutset_rows));
    appendVector(passive_eqs, matrixVecMulSubset(loop_matrix, u_vec, passive_loop_rows));
    if (verbose) {
        std::cout << "Passive equations:" << std::endl;
        print(passive_eqs);
    }
    auto [A, b] = SymEngine::linear_eqns_to_matrix(passive_eqs, passive_unknowns);
    std::vector<SymEngine::RCP<const SymEngine::Basic>> passive_solved = solve_linear_system_fast(A, b, passive_unknowns);

    std::unordered_map<std::string, SymEngine::RCP<const SymEngine::Basic>> solved;
    for (int i = 0; i < passive_unknowns.size(); ++i) {
        solved[passive_unknowns[i]->__str__()] = passive_solved[i];
    }

    assert(solved.size() == passive_unknowns.size());
    // Update passive_unknowns to output
    for (auto& [unknown, result] : solved) {
        if (verbose) {
            std::cout << "Solved " << unknown << " = " << result->__str__() << std::endl;
        }
        Component* comp = netlist.getComponent(unknown.substr(2));
        assert(comp != nullptr);
        if (unknown[0] == 'I') {
            output_currents[unknown] = result;
            comp->setCurrent(result);
        }
        if (unknown[0] == 'V') {
            output_voltages[unknown] = result;
            comp->setVoltage(result);
        }
        for (auto& src : netlist.vv_sources) {
            src->setVoltage(src->voltage().subs({{Expression(unknown), result}}));
        }
        for (auto& src : netlist.iv_sources) {
            src->setVoltage(src->voltage().subs({{Expression(unknown), result}}));
        }
        for (auto& src : netlist.ii_sources) {
            src->setCurrent(src->current().subs({{Expression(unknown), result}}));
        }
        for (auto& src : netlist.vi_sources) {
            src->setCurrent(src->current().subs({{Expression(unknown), result}}));
        }
    }
    // Update output
    for (Component* src : netlist.vv_sources) {
        output_voltages["V_" + src->name()] = src->voltage();
    }
    for (Component* src : netlist.iv_sources) {
        output_voltages["V_" + src->name()] = src->voltage();
    }
    for (Component* src : netlist.ii_sources) {
        output_currents["I_" + src->name()] = src->current();
    }
    for (Component* src : netlist.vi_sources) {
        output_currents["I_" + src->name()] = src->current();
    }
    for (Component* res : netlist.resistors) {
        output_voltages["V_" + res->name()] = res->voltage();
    }
    for (std::unique_ptr<Node>& node : netlist.nodes) {
        // Skip ground and non-output nodes
        if (node->name() == "0" || !contains(netlist.outputs, node->name())) {
            continue;
        }
        Expression node_voltage = nodeVoltage(node.get(), full_graph);
        output_voltages[node->name()] = node_voltage;
    }
    std::unordered_map<std::string, SymEngine::Expression> all_outputs;
    all_outputs.insert(output_currents.begin(), output_currents.end());
    all_outputs.insert(output_voltages.begin(), output_voltages.end());

    for (auto& mut : netlist.mutual_inductors) {
        // K * sqrt(L1 * L2)
        std::string K = std::get<0>(mut);
        Component* L1 = std::get<1>(mut);
        Component* L2 = std::get<2>(mut);
        Expression m = Expression(K) * SymEngine::sqrt(Expression(L1->value()) * L2->value());
        L1->addMutualInductance(m * L2->derivative());
        L2->addMutualInductance(m * L1->derivative());
    }

    /* Collect derivative equations */
    std::vector<SymEngine::RCP<const SymEngine::Basic>> deriv_eqs;
    appendVector(deriv_eqs, matrixVecMulSubset(cutset_matrix, i_vec, capacitor_cutset_rows));
    appendVector(deriv_eqs, matrixVecMulSubset(loop_matrix, u_vec, inductor_loop_rows));
    if (verbose) {
        std::cout << "Derivative Equations:" << std::endl;
        print(deriv_eqs);
    }

    /* Add dependent inductors and capacitors as states because depending on the switch combination
     * different inductors/capacitors can be the dependent components */
    for (Component* ind : dependent_inductors) {
        Expression current = Expression(std::format("I_{}", ind->name())) - ind->current();
        current = SymEngine::expand(Expression(str::replaceAll(current.get_basic()->__str__(), "I_", "dI_")));
        deriv_eqs.push_back(current);
        states.push_back(SymEngine::make_rcp<const Symbol>("I_" + ind->name()));
    }
    for (Component* cap : dependent_capacitors) {
        Expression voltage = Expression(std::format("V_{}", cap->name())) - cap->voltage();
        voltage = SymEngine::expand(Expression(str::replaceAll(voltage.get_basic()->__str__(), "V_", "dV_")));
        deriv_eqs.push_back(voltage);
        states.push_back(SymEngine::make_rcp<const Symbol>("V_" + cap->name()));
    }

    for (auto& eq : deriv_eqs) {
        for (auto& [name, sol] : solved) {
            eq = Expression(eq).subs({{Expression(name), Expression(sol)}});
        }
    }

    // Replace uL = L*di/dt and iC = C * du/dt
    for (auto& state : states) {
        Component* comp = netlist.getComponent(state->get_name().substr(2));
        if (comp->name()[0] == 'L') {
            replace(deriv_eqs, comp->voltage(), comp->v_derivative());
        } else if (comp->name()[0] == 'C') {
            replace(deriv_eqs, comp->current(), comp->i_derivative());
        }
    }
    if (verbose) {
        std::cout << "Derivative Equations:" << std::endl;
        print(deriv_eqs);
    }

    // Sort states
    std::sort(states.begin(), states.end(), [](auto const& a, auto const& b) {
        return a->get_name() < b->get_name();
    });

    std::vector<SymEngine::RCP<const Symbol>> states_deriv;
    for (auto const& state : states) {
        states_deriv.push_back(SymEngine::make_rcp<const Symbol>("d" + state->get_name()));
    }

    /* K1, A1, B1 */
    auto [K1, Bu1] = SymEngine::linear_eqns_to_matrix(deriv_eqs, states_deriv);
    assert(K1.nrows() == K1.ncols());
    auto [A1, Bu2] = SymEngine::linear_eqns_to_matrix(matrixToVector(Bu1), states);
    Bu2.mul_scalar(Expression("-1"), Bu2); // Multiplication by -1 to move to RHS
    std::vector<SymEngine::RCP<const SymEngine::Basic>> Bu2_eqs = matrixToVector(Bu2);
    std::vector<RCP<const Symbol>> inputs;
    for (auto const& input : netlist.inputs) {
        inputs.push_back(SymEngine::make_rcp<const Symbol>(input));
    }

    auto [B1, empty] = SymEngine::linear_eqns_to_matrix(Bu2_eqs, inputs);

    /* C1, D1 */
    std::vector<SymEngine::RCP<const SymEngine::Basic>> Cx_Du;
    for (int i = 0; i < netlist.outputs.size(); ++i) {
        Expression& eq = all_outputs.at(netlist.outputs.at(i));
        Cx_Du.push_back(eq);
    }
    std::vector<SymEngine::RCP<const Symbol>> states2;
    for (auto const& state : states) {
        if (state->get_name()[0] == 'V') {
            states2.push_back(SymEngine::make_rcp<const Symbol>("I_" + state->get_name().substr(2)));
        } else if (state->get_name()[0] == 'I') {
            states2.push_back(SymEngine::make_rcp<const Symbol>("V_" + state->get_name().substr(2)));
        } else {
            assert(("Invalid state variable", false));
        }
    }
    auto [C1, C2] = SymEngine::linear_eqns_to_matrix(Cx_Du, states);
    C2.mul_scalar(Expression("-1"), C2);
    auto [C3, C4] = SymEngine::linear_eqns_to_matrix(matrixToVector(C2), states2);
    C4.mul_scalar(Expression("-1"), C4);
    auto [D1, empty2] = SymEngine::linear_eqns_to_matrix(matrixToVector(C4), inputs);

    // H is diagonal matrix containing L and C vales
    SymEngine::DenseMatrix H1 = zeroMatrix((int)states.size(), (int)states.size());
    for (int i = 0; i < states.size(); ++i) {
        Component* component = netlist.getComponent(states[i]->get_name().substr(2));
        H1.set(i, i, SymEngine::Expression(component->value()));
    }
    SymEngine::DenseMatrix K2 = zeroMatrix((int)netlist.outputs.size(), (int)states.size());
    C3.mul_matrix(H1, K2);
    // Print results
    if (verbose) {
        std::cout << "K1 matrix:" << std::endl;
        std::cout << K1.__str__() << std::endl;
        std::cout << "K2 matrix:" << std::endl;
        std::cout << K2.__str__() << std::endl;
        std::cout << "A1 matrix:" << std::endl;
        std::cout << A1.__str__() << std::endl;
        std::cout << "B1 matrix:" << std::endl;
        std::cout << B1.__str__() << std::endl;
        std::cout << "C1 matrix:" << std::endl;
        std::cout << C1.__str__() << std::endl;
        std::cout << "D1 matrix:" << std::endl;
        std::cout << D1.__str__() << std::endl;
    }

    // Collect results
    StateSpaceMatrices matrices{
        .K1 = matrixToStr(K1),
        .K2 = matrixToStr(K2),
        .A1 = matrixToStr(A1),
        .B1 = matrixToStr(B1),
        .C1 = matrixToStr(C1),
        .D1 = matrixToStr(D1),
    };
    return matrices;
}

} // namespace rlc2ss
