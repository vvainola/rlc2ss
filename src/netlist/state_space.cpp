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
#include "Eigen/Core"
#pragma warning(pop)

#include <iostream>

namespace rlc2ss {

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

void replace(std::vector<LinearExpr>& vec, std::string const& old_symbol, LinearExpr const& new_expr) {
    for (auto& expr : vec) {
        expr.replace(old_symbol, new_expr);
    }
}

void print(std::vector<LinearExpr> const& exprs) {
    for (auto& expr : exprs) {
        std::cout << expr.str() << std::endl;
    }
}

std::string matrixToStr(Eigen::MatrixXd const& matrix) {
    std::string result;
    for (unsigned int i = 0; i < matrix.rows(); ++i) {
        for (unsigned int j = 0; j < matrix.cols(); ++j) {
            if (i != matrix.rows() - 1 || j != matrix.cols() - 1) {
                result += std::format("{},", matrix(i, j));
            } else {
                result += std::format("{}", matrix(i, j));
            }
        }
    }
    return result;
}

std::vector<LinearExpr> getRows(std::vector<LinearExpr> const& vec, std::vector<int> const& rows) {
    std::vector<LinearExpr> result;
    for (unsigned int i = 0; i < rows.size(); ++i) {
        result.push_back(vec[rows[i]]);
    }
    return result;
}

Eigen::MatrixXd getRows(Eigen::MatrixXd const& matrix, std::vector<int> const& rows) {
    Eigen::MatrixXd out_matrix(rows.size(), matrix.cols());
    for (int row = 0; row < rows.size(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            out_matrix(row, col) = matrix(rows[row], col);
        }
    }
    return out_matrix;
}

LinearExpr nodeVoltage(Node const* node, Graph const& graph) {
    // Traverse from ground to node
    LinearExpr node_voltage = 0;
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

StateSpaceMatrices formStateSpaceMatrices(std::string const& netlist_str,
                                          uint64_t combination,
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
    std::vector<std::string> states;
    std::vector<Component*> dependent_capacitors;
    std::vector<Component*> dependent_inductors;
    std::unordered_map<std::string, LinearExpr> output_currents;
    std::unordered_map<std::string, LinearExpr> output_voltages;
    for (Component* cap : netlist.capacitors) {
        output_voltages[cap->voltage().str()] = cap->voltage();
        output_currents[cap->current().str()] = cap->current();
        if (contains(twigs, cap)) {
            states.push_back(cap->voltage().str());
        } else {
            dependent_capacitors.push_back(cap);
        }
    }
    for (Component* ind : netlist.inductors) {
        output_voltages[ind->voltage().str()] = ind->voltage();
        output_currents[ind->current().str()] = ind->current();
        if (!contains(twigs, ind)) {
            states.push_back(ind->current().str());
        } else {
            dependent_inductors.push_back(ind);
        }
    }

    // Create cutset matrix
    std::unordered_map<std::string, LinearExpr> solved;
    Eigen::MatrixXd cutset_matrix = Eigen::MatrixXd::Zero((int)twigs.size(), (int)netlist.components.size());
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
                    cutset_matrix(i, netlist.getComponentIndex(link->name())) = 1;
                } else {
                    cutset_matrix(i, netlist.getComponentIndex(link->name())) = -1;
                }
            }
        }
        cutset_matrix(i, netlist.getComponentIndex(twig->name())) = 1;

        // Add twig back
        proper_tree.addComponent(twig);

        if (twig->name()[0] == 'C') {
            capacitor_cutset_rows.push_back(i);
        } else if (twig->name()[0] == 'L') {
            dep_inductor_cutset_rows.push_back(i);
        } else if (twig->name()[0] == 'I') {
            //std::cout << std::format("Bad cutset: current source {} in twig! Setting to zero.", twig->name()) << std::endl;
            solved[twig->voltage().str()] = 0;
            output_voltages[twig->voltage().str()] = 0;
        } else {
            passive_cutset_rows.push_back(i);
        }
    }

    // Create loop matrix
    Eigen::MatrixXd loop_matrix = Eigen::MatrixXd::Zero((int)links.size(), (int)netlist.components.size());
    std::vector<int> inductor_loop_rows;
    std::vector<int> dep_capacitor_loop_rows;
    std::vector<int> passive_loop_rows;
    for (int i = 0; i < links.size(); ++i) {
        Component* link = links[i];
        std::vector<Node*> path = proper_tree.dijkstra(link->posNode(), link->negNode());
        for (int j = 1; j < path.size(); ++j) {
            Node* pos_node = path[j - 1];
            Node* neg_node = path[j];
            Component* comp = proper_tree.getComponent(pos_node, neg_node);
            if (comp->posNode()->name() == pos_node->name()) {
                loop_matrix(i, netlist.getComponentIndex(comp->name())) = -1;
            } else {
                loop_matrix(i, netlist.getComponentIndex(comp->name())) = 1;
            }
        }
        Node* pos_node = path[path.size() - 1];
        if (link->posNode()->name() == pos_node->name()) {
            loop_matrix(i, netlist.getComponentIndex(link->name())) = -1;
        } else {
            loop_matrix(i, netlist.getComponentIndex(link->name())) = 1;
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
        LinearExpr pos_node_voltage = nodeVoltage(src->posSource(), full_graph);
        LinearExpr neg_node_voltage = nodeVoltage(src->negSource(), full_graph);
        if (src->name()[0] == 'E') {
            src->setVoltage(src->value() * (pos_node_voltage - neg_node_voltage));
        } else if (src->name()[0] == 'G') {
            src->setCurrent(src->value() * (pos_node_voltage - neg_node_voltage));
        }
        full_graph.addComponent(src);
    }

    // Form voltage and current vectors
    std::vector<LinearExpr> i_vec((int)netlist.components.size(), 0);
    std::vector<LinearExpr> u_vec((int)netlist.components.size(), 0);
    for (int i = 0; i < netlist.components.size(); ++i) {
        i_vec[i] = netlist.components[i]->current();
        u_vec[i] = netlist.components[i]->voltage();
    }

    // Collect equations corresponding to dependent components
    std::vector<LinearExpr> cutset_dep_eqs = getRows(cutset_matrix, dep_inductor_cutset_rows) * i_vec;
    std::vector<LinearExpr> loop_dep_eqs = getRows(loop_matrix, dep_capacitor_loop_rows) * u_vec;
    std::vector<LinearExpr> dep_eqs;
    appendVector(dep_eqs, cutset_dep_eqs);
    appendVector(dep_eqs, loop_dep_eqs);
    if (verbose) {
        std::cout << "Dependent equations:" << std::endl;
        print(dep_eqs);
    }

    // Replace currents in inductors which are not state variables with state currents
    // Replace voltages in capacitors which are not state variables with state voltages
    std::vector<std::string> dep_unknowns;
    for (Component* ind : dependent_inductors) {
        dep_unknowns.push_back(ind->current().str());
    }
    for (Component* cap : dependent_capacitors) {
        dep_unknowns.push_back(cap->voltage().str());
    }
    if (dep_unknowns.size() > 0) {
        auto [A, b] = linearEqsToMatrix(dep_eqs, dep_unknowns);
        std::vector<LinearExpr> dep_solved = solveLinearSystem(A, b, dep_unknowns);
        assert(dep_solved.size() == dep_unknowns.size());
        for (unsigned int i = 0; i < dep_unknowns.size(); ++i) {
            std::string dep_unknown = dep_unknowns[i];
            Component* comp = netlist.getComponent(dep_unknown.substr(2));
            if (dep_unknown[0] == 'I') {
                comp->setCurrent(dep_solved[i]);
                output_currents[dep_unknown] = dep_solved[i];
            } else if (dep_unknown[0] == 'V') {
                comp->setVoltage(dep_solved[i]);
                output_voltages[dep_unknown] = dep_solved[i];
            }
            replace(i_vec, dep_unknown, dep_solved[i]);
            replace(u_vec, dep_unknown, dep_solved[i]);
        }
    }

    /* Replace the passive components currents and voltages with state variables. */
    std::vector<std::string> passive_unknowns;
    for (auto& current : netlist.current_sources) {
        if (!solved.contains(current->voltage().str())) {
            passive_unknowns.push_back(current->voltage().str());
        }
    }
    for (auto& current : netlist.ii_sources) {
        passive_unknowns.push_back(current->voltage().str());
    }
    for (auto& current : netlist.vi_sources) {
        passive_unknowns.push_back(current->voltage().str());
    }
    for (auto& voltage : netlist.voltage_sources) {
        passive_unknowns.push_back(voltage->current().str());
    }
    for (auto& voltage : netlist.vv_sources) {
        passive_unknowns.push_back(voltage->current().str());
    }
    for (auto& voltage : netlist.iv_sources) {
        passive_unknowns.push_back(voltage->current().str());
    }
    for (auto& resistor : netlist.resistors) {
        passive_unknowns.push_back(resistor->current().str());
    }

    // Collect equations corresponding to passive components
    std::vector<LinearExpr> cutset_passive_eqs = getRows(cutset_matrix, passive_cutset_rows) * i_vec;
    std::vector<LinearExpr> loop_passive_eqs = getRows(loop_matrix, passive_loop_rows) * u_vec;
    std::vector<LinearExpr> passive_eqs;
    appendVector(passive_eqs, cutset_passive_eqs);
    appendVector(passive_eqs, loop_passive_eqs);
    if (verbose) {
        std::cout << "Passive equations:" << std::endl;
        print(passive_eqs);
    }
    auto [A, b] = linearEqsToMatrix(passive_eqs, passive_unknowns);
    std::vector<LinearExpr> passive_solved = solveLinearSystem(A, b, passive_unknowns);

    for (int i = 0; i < passive_unknowns.size(); ++i) {
        solved[passive_unknowns[i]] = passive_solved[i];
        replace(i_vec, passive_unknowns[i], passive_solved[i]);
        replace(u_vec, passive_unknowns[i], passive_solved[i]);
    }

    // Update passive_unknowns to output
    for (auto& [unknown, result] : solved) {
        if (verbose) {
            std::cout << "Solved " << unknown << " = " << result.str() << std::endl;
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
            LinearExpr voltage = src->voltage();
            voltage.replace(unknown, result);
            src->setVoltage(voltage);
        }
        for (auto& src : netlist.iv_sources) {
            LinearExpr voltage = src->voltage();
            voltage.replace(unknown, result);
            src->setVoltage(voltage);
        }
        for (auto& src : netlist.ii_sources) {
            LinearExpr current = src->current();
            current.replace(unknown, result);
            src->setCurrent(current);
        }
        for (auto& src : netlist.vi_sources) {
            LinearExpr current = src->current();
            current.replace(unknown, result);
            src->setCurrent(current);
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
        LinearExpr node_voltage = nodeVoltage(node.get(), full_graph);
        output_voltages[node->name()] = node_voltage;
    }
    std::unordered_map<std::string, LinearExpr> all_outputs;
    all_outputs.insert(output_currents.begin(), output_currents.end());
    all_outputs.insert(output_voltages.begin(), output_voltages.end());

    for (auto& mut : netlist.mutual_inductors) {
        // Mutual inductance M = K * sqrt(L1 * L2). The voltage term added to L1
        // is M * dI_L2/dt, where derivativeSymbol() returns dI_L2 directly
        // (without the L2 value factor that derivative() applies).
        double K = std::stod(std::get<0>(mut));
        Component* L1 = std::get<1>(mut);
        Component* L2 = std::get<2>(mut);
        double m = K * sqrt(L1->value() * L2->value());
        L1->addMutualInductance(m * L2->derivativeSymbol());
        L2->addMutualInductance(m * L1->derivativeSymbol());
    }

    /* Collect derivative equations */
    std::vector<LinearExpr> deriv_eqs;
    std::vector<LinearExpr> cutset_deriv_eqs = getRows(cutset_matrix, capacitor_cutset_rows) * i_vec;
    std::vector<LinearExpr> loop_deriv_eqs = getRows(loop_matrix, inductor_loop_rows) * u_vec;
    appendVector(deriv_eqs, cutset_deriv_eqs);
    appendVector(deriv_eqs, loop_deriv_eqs);
    if (verbose) {
        std::cout << "Derivative Equations:" << std::endl;
        print(deriv_eqs);
    }

    /* Add dependent inductors and capacitors as states because depending on the switch combination
     * different inductors/capacitors can be the dependent components */
    for (Component* ind : dependent_inductors) {
        LinearExpr current = LinearExpr(std::format("I_{}", ind->name())) - ind->current();
        current.replace("I_" + ind->name(), "dI_" + ind->name());
        for (auto& state : states) {
            if (state[0] != 'I') {
                continue;
            }
            current.replace(state, "d" + state);
        }
        deriv_eqs.push_back(current);
        states.push_back("I_" + ind->name());
    }
    for (Component* cap : dependent_capacitors) {
        LinearExpr voltage = LinearExpr(std::format("V_{}", cap->name())) - cap->voltage();
        voltage.replace("V_" + cap->name(), "dV_" + cap->name());
        for (auto& state : states) {
            if (state[0] != 'V') {
                continue;
            }
            voltage.replace(state, "d" + state);
        }

        deriv_eqs.push_back(voltage);
        states.push_back("V_" + cap->name());
    }

    for (auto& eq : deriv_eqs) {
        for (auto& [name, sol] : solved) {
            eq.replace(name, sol);
        }
    }

    // Replace uL = L*di/dt and iC = C * du/dt
    for (auto& state : states) {
        Component* comp = netlist.getComponent(state.substr(2));
        if (comp->name()[0] == 'L') {
            assert(comp->voltage().terms.size() == 1);
            assert(comp->voltage().terms.begin()->second == 1);
            assert(comp->voltage().constant == 0);
            assert(comp->voltage().str() == "V_" + comp->name());
            replace(deriv_eqs, comp->voltage().str(), comp->v_derivative());
        } else if (comp->name()[0] == 'C') {
            assert(comp->current().terms.size() == 1);
            assert(comp->current().terms.begin()->second == 1);
            assert(comp->current().constant == 0);
            assert(comp->current().str() == "I_" + comp->name());
            replace(deriv_eqs, comp->current().str(), comp->i_derivative());
        }
    }
    if (verbose) {
        std::cout << "Derivative Equations:" << std::endl;
        print(deriv_eqs);
    }

    // Sort states
    std::sort(states.begin(), states.end());
    std::vector<std::string> inputs = netlist.inputs;
    std::sort(inputs.begin(), inputs.end());

    std::vector<std::string> states_deriv;
    for (auto const& state : states) {
        states_deriv.push_back("d" + state);
    }

    /* K1, A1, B1 */
    auto [K1, Bu1] = linearEqsToMatrix(deriv_eqs, states_deriv);
    auto [A1, Bu2] = linearEqsToMatrix(Bu1, states);
    for (auto& expr : Bu2) {
        expr = -expr; // Multiplication by -1 to move to RHS
    }
    auto [B1, empty] = linearEqsToMatrix(Bu2, inputs);

    /* C1, D1 */
    std::vector<LinearExpr> Cx_Du;
    for (int i = 0; i < netlist.outputs.size(); ++i) {
        LinearExpr& eq = all_outputs.at(netlist.outputs.at(i));
        Cx_Du.push_back(eq);
    }
    std::vector<std::string> states2;
    for (auto const& state : states) {
        if (state[0] == 'V') {
            states2.push_back(("I_" + state.substr(2)));
        } else if (state[0] == 'I') {
            states2.push_back(("V_" + state.substr(2)));
        } else {
            assert(("Invalid state variable", false));
        }
    }
    auto [C1, C2] = linearEqsToMatrix(Cx_Du, states);
    for (auto& expr : C2) {
        expr = -expr; // Multiplication by -1 to move to RHS
    }
    auto [C3, C4] = linearEqsToMatrix(C2, states2);
    for (auto& expr : C4) {
        expr = -expr; // Multiplication by -1 to move to RHS
    }
    auto [D1, empty2] = linearEqsToMatrix(C4, inputs);

    // H is diagonal matrix containing L and C vales
    Eigen::MatrixXd H1 = Eigen::MatrixXd::Zero((int)states.size(), (int)states.size());
    for (int i = 0; i < states.size(); ++i) {
        Component* component = netlist.getComponent(states[i].substr(2));
        H1(i, i) = component->value();
    }
    Eigen::MatrixXd K2 = C3 * H1;
    // Print results
    if (verbose) {
        std::cout << "K1 matrix:" << std::endl;
        std::cout << K1 << std::endl;
        std::cout << "K2 matrix:" << std::endl;
        std::cout << K2 << std::endl;
        std::cout << "A1 matrix:" << std::endl;
        std::cout << A1 << std::endl;
        std::cout << "B1 matrix:" << std::endl;
        std::cout << B1 << std::endl;
        std::cout << "C1 matrix:" << std::endl;
        std::cout << C1 << std::endl;
        std::cout << "D1 matrix:" << std::endl;
        std::cout << D1 << std::endl;
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
