# MIT License
#
# Copyright (c) 2022 vvainola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Form state-space matrices of an RLC circuit from a netlist using proper tree
# method explained in http://circuit-simulation.net/equation_generation.html

import os
import sympy as sy
from sympy import Symbol, Matrix
import typing as T
import networkx as nx
import sys
import state_matrices_to_json
from state_matrices_to_json import StateSpaceMatrices, Diode
import itertools
import argparse
from tqdm import tqdm

sy.init_printing()
M = Matrix
S = Symbol

class Node:
    def __init__(self, name: str):
        self.name: str = name
        self.connections: T.List[Component] = []

    def __eq__(self, o: object) -> bool:
        return self.name == o


class Component:
    def __init__(self, name: str, pos_node: Node, neg_node: Node, default_value: str):
        self.name: str = name
        self.pos_node: Node = pos_node
        self.neg_node: Node = neg_node
        self.pos_src: T.Optional[Symbol] = None
        self.neg_src: T.Optional[Symbol] = None
        if default_value.endswith("N"):
            self.default_value = float(default_value[:-1]) * 1e-9
        elif default_value.endswith("U"):
            self.default_value = float(default_value[:-1]) * 1e-6
        elif default_value.endswith("M"):
            self.default_value = float(default_value[:-1]) * 1e-3
        else:
            try:
                self.default_value = float(default_value)
            except (ValueError, TypeError):
                self.default_value = -1.0

        if name[0] == 'V':
            self._voltage = Symbol(self.name)
            self._current = Symbol(f'I_{self.name}')
        elif name[0] == 'I':
            self._voltage = Symbol(f'V_{self.name}')
            self._current = Symbol(f'{self.name}')
        elif name[0] == 'R':
            self._voltage = Symbol(self.name) * Symbol(f'I_{self.name}')
            self._current = Symbol(f'I_{self.name}')
        else:
            self._voltage = Symbol(f'V_{self.name}')
            self._current = Symbol(f'I_{self.name}')

        if self.name[0] == 'C':
            self._der = Symbol(f'd{self.v()}')
        elif self.name[0] == 'L':
            self._der =  Symbol(f'd{self.i()}')
        else:
            self._der = None

        if name.startswith('I_switch'):
            self._current = 0
        elif name.startswith('V_switch'):
            self._voltage = 0

        self._mutual_inductance_voltage = sy.Add()

    def __str__(self) -> str:
        return self.name

    def v(self):
        return self._voltage

    def i(self):
        return self._current

    def der(self):
        return self._der

    def v_der(self):
        if self.name[0] != 'L':
            assert False, f"Tried to get Ldi/dt term for {self.name}"
        return Symbol(self.name) * self.der() + self._mutual_inductance_voltage

    def i_der(self):
        if self.name[0] != 'C':
            assert False, f"Tried to get Cdu/dt term for {self.name}"
        return Symbol(self.name) * self.der()

    def update_current(self, new_current):
        self._current = new_current
        if self.name[0] == 'L':
            self._der = sy.sympify(str(new_current).replace('I_', 'dI_'))
        elif self.name[0] == 'R':
            self._voltage = self._voltage = Symbol(self.name) * self._current

    def update_voltage(self, new_voltage):
        self._voltage = new_voltage
        if self.name[0] == 'C':
            self._der = sy.sympify(str(new_voltage).replace('V_', 'dV_'))

    def update_der(self, new_der):
        if self.name[0] != 'C' or self.name[0] != 'L':
            assert False, f"Tried to set derivative term for {self.name}"
        self._der = new_der

    def update_src(self, pos, neg = None):
        self.pos_src = pos
        self.neg_src = neg

    def add_mutual_inductance(self, mutual_inductance_voltage):
        self._mutual_inductance_voltage += mutual_inductance_voltage


def parse_netlist(filename):
    netlist = open(filename, 'r').readlines()

    parsed_netlist = []
    control_section = False
    for line in netlist:
        line = line.strip()
        if line.startswith(".control"):
            control_section = True
        # Leave out
        # - lines in control section
        # - Empty lines
        # - Comment lines . ; *
        if not (control_section) and line != '' and not (line[0] in ['.', '*', ';', '#']):
            # Capitalize each line
            # Remove extra spaces between elements
            line = ' '.join(line.split())
            line += ' '
            parsed_netlist.append(line)
        if line.startswith(".endc"):
            control_section = False
    return parsed_netlist

def replace_diodes(netlist : list[str]) -> tuple[list[str], list[Diode]]:
    new_netlist = []
    diodes: list[Diode] = []
    for line in netlist:
        if line.startswith('D'):
            line_split = line.split()
            name = line_split[0]
            pos_node = line_split[1]
            neg_node = line_split[2]
            new_netlist.append(f'V_{name} {pos_node} _N_{name}_1 Vp;')
            new_netlist.append(f'S_{name} _N_{name}_1 _N_{name}_2')
            new_netlist.append(f'R_{name} _N_{name}_2 {neg_node} Vn;I;')
            diodes.append(
                Diode(
                    name= name,
                    pos_node=pos_node,
                    neg_node=neg_node,
                    forward_voltage=f'V_{name}',
                    current=f'I_R_{name}',
                    switch=f'S_{name}'
                )
            )
        else:
            new_netlist.append(line)
    return (new_netlist, diodes)

def get_component(all_components, component_name):
    for c in all_components:
        if c.name == component_name:
            return c
    assert False, "Tried to access non-existent component"

def lines_with_switches(netlist):
    lines_w_switches = []
    switches = []
    xor_switches = []
    and_switches = []
    for line in netlist:
        line_split = line.strip().split(' ')
        if line_split[0][0] == 'S':
            lines_w_switches.append(line)
            switches.append(line_split[0])
        if line_split[0][0] == 'X':
            xor_switch_combination = line_split[1:]
            xor_switches.append(xor_switch_combination)
            # Check for invalid switches
            if set(switches).intersection(set(xor_switch_combination)) != set(xor_switch_combination):
                sys.exit(f'Invalid switch in {xor_switch_combination}')
        if line_split[0][0] == 'Y':
            and_switch_combination = line_split[1:]
            and_switches.append(and_switch_combination)
            # Check for invalid switches
            if set(switches).intersection(set(and_switch_combination)) != set(and_switch_combination):
                sys.exit(f'Invalid switch in {and_switch_combination}')
    lines_w_switches.sort()
    switches.sort()
    return lines_w_switches, switches, xor_switches, and_switches

def is_invalid_switch_combination(combination, switches, xor_switches, and_switches):
    active_switches = set()
    for i, v in enumerate(combination):
        if v == 1:
            active_switches.add(switches[i])
    for xor_combination in xor_switches:
        if len(active_switches.intersection(set(xor_combination))) > 1:
            return True
    for and_combination in and_switches:
        count = len(active_switches.intersection(set(and_combination)))
        if count >= 1 and count < len(and_combination):
            return True
    return False

def get_xv_voltage(vv_src : Component, components : T.List[Component], full_tree):
    full_tree = full_tree.copy()
    full_tree.remove_edge(vv_src.pos_node.name, vv_src.neg_node.name)
    voltages = []
    for node in [vv_src.pos_src, vv_src.neg_src]:
        if node.name == '0':
            voltages.append(Symbol('0'))
            continue
        path = list(nx.shortest_path(full_tree, '0', node.name))
        node_voltage = sy.Add()
        for j in range(1, len(path)):
            pos = path[j - 1]
            neg = path[j]
            comp = components[full_tree[pos][neg]['edge_idx']]
            if comp.pos_node.name == pos:
                node_voltage -= comp.v()
            else:
                node_voltage += comp.v()
        voltages.append(node_voltage)
    return Symbol(vv_src.name) * (voltages[0] - voltages[1])

# @profile
def form_state_space_matrices(parsed_netlist) -> StateSpaceMatrices:
    NAME = 0
    POS_NODE = 1
    NEG_NODE = 2
    DEP_V_SRC_POS = 3
    DEP_V_SRC_NEG = 4
    DEP_I_SRC = 3
    netlist = parsed_netlist
    nodes: T.List[Node] = []
    components: T.List[Component] = []
    voltage_sources: T.List[Component] = []
    current_sources: T.List[Component] = []
    resistors: T.List[Component] = []
    capacitors: T.List[Component] = []
    inductors: T.List[Component] = []
    outputs: T.List[str] = []
    vv_sources: T.List[Component] = []
    iv_sources: T.List[Component] = []
    vi_sources: T.List[Component] = []
    ii_sources: T.List[Component] = []
    mutual_inductors: T.List[T.Tuple[str, Component, Component]] = []
    for line in netlist:
        line_split = line.split()
        if line_split[NAME][0] == 'K' or line_split[NAME][0] == 'X' or line_split[NAME][0] == 'Y':
            continue
        node_pos = Node(line_split[POS_NODE])
        node_neg = Node(line_split[NEG_NODE])
        if node_pos not in nodes:
            nodes.append(node_pos)
        if node_neg not in nodes:
            nodes.append(node_neg)
    for line in netlist:
        line_split = line.split()
        if line_split[NAME][0] == 'K':
            L1 = get_component(components, line_split[1])
            L2 = get_component(components, line_split[2])
            mutual_inductors.append((line_split[NAME], L1, L2))
            continue
        elif line_split[NAME][0] == 'X' or line_split[NAME][0] == 'Y':
            continue

        pos_node = nodes[nodes.index(Node(line_split[POS_NODE]))]
        neg_node = nodes[nodes.index(Node(line_split[NEG_NODE]))]
        name = line_split[NAME]
        if name.startswith('V') or name.startswith('I'):
            default_value_txt = line_split[4] if len(line_split) > 4 else "-1"
        elif name.startswith('R') or name.startswith('L') or name.startswith('C'):
            default_value_txt = line_split[3] if len(line_split) > 3 else "-1"
        else:
            default_value_txt = "-1" # Ignore default
        # The default value may have output specifiers after ;
        default_value = default_value_txt.split(';')[0]
        component = Component(line_split[NAME], pos_node, neg_node, default_value)
        neg_node.connections.append(component)
        pos_node.connections.append(component)
        components.append(component)

        if component.name[0] == 'V':
            voltage_sources.append(component)
        elif component.name[0] == 'I':
            current_sources.append(component)
        elif component.name[0] == 'R':
            resistors.append(component)
        elif component.name[0] == 'L':
            inductors.append(component)
            outputs.append((f'I_{component.name}'))
        elif component.name[0] == 'C':
            capacitors.append(component)
            outputs.append((f'V_{component.name}'))
        elif component.name[0] == 'E':
            vv_sources.append(component)
            pos = Symbol(line_split[DEP_V_SRC_POS])
            neg = Symbol(line_split[DEP_V_SRC_NEG])
            component.update_src(pos, neg)
            component.update_voltage(Symbol(line_split[NAME]) * (pos - neg))
        elif component.name[0] == 'F':
            ii_sources.append(component)
            src = Symbol(f'I_{line_split[DEP_I_SRC]}')
            component.update_src(src)
            component.update_current(Symbol(line_split[NAME]) * src)
        elif component.name[0] == 'G':
            vi_sources.append(component)
            pos = Symbol(line_split[DEP_V_SRC_POS])
            neg = Symbol(line_split[DEP_V_SRC_NEG])
            component.update_src(pos, neg)
            component.update_current(Symbol(line_split[NAME]) * (pos - neg))
        elif component.name[0] == 'H':
            iv_sources.append(component)
            src = Symbol(f'I_{line_split[DEP_I_SRC]}')
            component.update_src(src)
            component.update_voltage(Symbol(line_split[NAME]) * src)
        else:
            assert False, f"Unknown component type {component.name}"

        # Parse possible extra outputs from default value e.g. "50e-3;I;Vc;"
        comp_outputs = default_value_txt.upper()
        if pos_node.name.startswith('N'):
            outputs.append((pos_node.name))
        if neg_node.name.startswith('N'):
            outputs.append((neg_node.name))
        if 'VP;' in comp_outputs:
            outputs.append((pos_node.name))
        if 'VN;' in comp_outputs:
            outputs.append((neg_node.name))
        if 'VC;' in comp_outputs:
            if component.name[0] == 'V':
                outputs.append((component.name))
            else:
                outputs.append((f'V_{component.name}'))
        if 'I;' in comp_outputs:
            if component.name[0] == 'I':
                outputs.append((component.name))
            else:
                outputs.append((f'I_{component.name}'))
    outputs = list(set(outputs))
    outputs.sort()
    outputs = [Symbol(output) for output in outputs]
    if outputs[0] == Symbol('0'):
        outputs = outputs[1:]

    temp_tree = nx.Graph()
    for c in components:
        temp_tree.add_edge(c.pos_node.name, c.neg_node.name, edge_idx=components.index(c))

    ground = nodes[nodes.index(Node('0'))]
    for node in nodes:
        if not nx.has_path(temp_tree, node.name, ground.name):
            component = Component(f'V_switch_{node.name}', node, ground, '0')
            voltage_sources.append(component)
            components.append(component)
            temp_tree.add_edge(node.name, ground.name, edge_idx=components.index(component))
            node.connections.append(component)
            ground.connections.append(component)

    # Select nodes belonging to the "proper tree"
    # All branches corresponding to a voltage source must be selected.
    # The maximum possible number of branches corresponding to a capacitor should be selected. Recall that the definition of a tree prohibits graph loops and as such, it may not be possible to include every capacitor of the network in the proper tree.
    # The maximum possible number of branches corresponding to a resistor should be selected such that the definition of tree is not violated.
    # The necessary number of branches corresponding to inductors and current sources required to complete the graph tree should be selected.
    twigs: T.List[Component] = []
    links: T.List[Component] = []
    proper_tree = nx.Graph()
    full_tree = nx.Graph()
    for node in nodes:
        proper_tree.add_node(node.name)
        full_tree.add_node(node.name)

    for comp in capacitors + resistors + inductors + current_sources + ii_sources + vi_sources:
        full_tree.add_edge(comp.pos_node.name, comp.neg_node.name, edge_idx=components.index(comp))

    for src in voltage_sources + vv_sources + iv_sources:
        twigs.append(src)
        proper_tree.add_edge(src.pos_node.name, src.neg_node.name, edge_idx=components.index(src))
        full_tree.add_edge(src.pos_node.name, src.neg_node.name, edge_idx=components.index(src))
    for comp in capacitors + resistors + inductors + current_sources + ii_sources + vi_sources:
        if not nx.has_path(proper_tree, comp.pos_node.name, comp.neg_node.name):
            twigs.append(comp)
            proper_tree.add_edge(comp.pos_node.name, comp.neg_node.name, edge_idx=components.index(comp))
        else:
            links.append(comp)

    # State variables are capacitors that are in twigs and inductors that are in branches
    dependent_voltages: T.List[sy.Symbol] = []
    dependent_currents: T.List[sy.Symbol] = []
    states: T.List[sy.Symbol] = []

    output_currents = {}
    output_voltages = {}
    for cap in capacitors:
        output_voltages[cap.v()] = cap.v()
        output_currents[cap.i()] = cap.i()
        if cap in twigs:
            states.append(cap.v())
        else:
            dependent_voltages.append(cap.v())
    for ind in inductors:
        output_voltages[ind.v()] = ind.v()
        output_currents[ind.i()] = ind.i()
        if ind not in twigs:
            states.append(ind.i())
        else:
            dependent_currents.append(ind.i())

    branches = len(components)
    # Create cutset matrix
    cutset_matrix = sy.zeros(len(twigs), branches)
    capacitor_cutset_rows = []
    dep_inductor_cutset_rows = []
    passive_cutset_rows = []
    solved = {}
    for i, twig in enumerate(twigs):
        # Remove twig
        proper_tree.remove_edge(twig.pos_node.name, twig.neg_node.name)
        for link in links:
            if not nx.has_path(proper_tree, link.pos_node.name, link.neg_node.name):
                if nx.has_path(proper_tree, twig.pos_node.name, link.pos_node.name):
                    cutset_matrix[i, components.index(link)] = 1
                else:
                    cutset_matrix[i, components.index(link)] = -1
        cutset_matrix[i, components.index(twig)] = 1
        # Add twig back
        proper_tree.add_edge(twig.pos_node.name, twig.neg_node.name, edge_idx=components.index(twig))

        if twig.name[0] == 'C':
            capacitor_cutset_rows.append(i)
        elif twig.name[0] == 'L':
            dep_inductor_cutset_rows.append(i)
        elif twig.name[0] == 'I':
            print(f"Bad cutset for current source {twig.name}. Setting to 0.")
            solved[f'V_{twig.name}'] = Symbol('0')
        else:
            passive_cutset_rows.append(i)
    # Create loop matrix
    inductor_loop_rows = []
    dep_capacitor_loop_rows = []
    passive_loop_rows = []
    loop_matrix = sy.zeros(len(links), branches)
    for i, link in enumerate(links):
        path = list(nx.shortest_path(proper_tree, link.pos_node.name, link.neg_node.name))
        for j in range(1, len(path)):
            pos = path[j - 1]
            neg = path[j]
            comp = components[proper_tree[pos][neg]['edge_idx']]
            if comp.pos_node.name == pos:
                loop_matrix[i, components.index(comp)] = -1
            else:
                loop_matrix[i, components.index(comp)] = 1
        pos = path[-1]
        if link.pos_node.name == pos:
            loop_matrix[i, components.index(link)] = -1
        else:
            loop_matrix[i, components.index(link)] = 1

        if link.name[0] == 'C':
            dep_capacitor_loop_rows.append(i)
        elif link.name[0] == 'L':
            inductor_loop_rows.append(i)
        else:
            passive_loop_rows.append(i)

    for src in vv_sources:
        src.update_voltage(get_xv_voltage(src, components, full_tree))
    for src in vi_sources:
        src.update_current(get_xv_voltage(src, components, full_tree))

    # Form voltage and current vectors
    i_vec = sy.zeros(branches, 1)
    u_vec = sy.zeros(branches, 1)
    for i, comp in enumerate(components):
        i_vec[i] = comp.i()
        u_vec[i] = comp.v()
    cutset_eqs = cutset_matrix * i_vec
    loop_eqs = loop_matrix * u_vec

    # Replace currents in inductors which are not state variables with state currents
    # Replace voltages in capacitors which are not state variables with state voltages
    dep_state_eqs =  [loop_eqs[i] for i in dep_capacitor_loop_rows] + [cutset_eqs[i] for i in dep_inductor_cutset_rows]
    dep_unknowns = dependent_currents + dependent_voltages
    if len(dep_unknowns) > 0:
        assert len(dep_state_eqs) == len(dep_unknowns), "Number of dependent state equations and unknowns do not match"
        for i, sol in enumerate(sy.linsolve(dep_state_eqs, dep_unknowns).args[0]):
            comp_name = str(dep_unknowns[i])[2:]
            comp = get_component(components, comp_name)
            if str(dep_unknowns[i])[0] == 'I':
                output_currents[dep_unknowns[i]] = sol
                comp.update_current(sol)
            else:
                output_voltages[dep_unknowns[i]] = sol
                comp.update_voltage(sol)
            solved[dep_unknowns[i]] = sol
            i_vec = i_vec.xreplace({dep_unknowns[i] : sol})
            u_vec = u_vec.xreplace({dep_unknowns[i] : sol})
        cutset_eqs = cutset_matrix * i_vec
        loop_eqs = loop_matrix * u_vec

    # Solve the passive components currents and voltages with state variables
    unknowns: T.List[Component] = []
    for comp in current_sources + ii_sources + vi_sources:
        if str(comp.v()) not in solved:
            unknowns.append(comp.v())
    for comp in resistors + voltage_sources + vv_sources + iv_sources:
        if str(comp.i()) not in solved:
            unknowns.append(comp.i())
    passive_eqs = [loop_eqs[i] for i in passive_loop_rows] + [cutset_eqs[i] for i in passive_cutset_rows]
    for i, sol in enumerate(sy.linsolve(passive_eqs, unknowns).args[0]):
        solved[unknowns[i]] = sol

    # Update solved to output
    for unknown, unknown_as_state_vars in solved.items():
        comp = get_component(components, str(unknown)[2:])
        if str(unknown)[0] == 'I':
            output_currents[unknown] = unknown_as_state_vars
            comp.update_current(unknown_as_state_vars)
        if str(unknown)[0] == 'V':
            output_voltages[unknown] = unknown_as_state_vars
            comp.update_voltage(unknown_as_state_vars)
        for src in vv_sources + iv_sources:
            src.update_voltage(src.v().xreplace({unknown : unknown_as_state_vars}))
        for src in ii_sources + vi_sources:
            src.update_current(src.i().xreplace({unknown : unknown_as_state_vars}))

    # Update outputs
    for src in vv_sources + iv_sources:
        output_voltages[Symbol(f'V_{src.name}')] = src.v()
    for src in ii_sources + vi_sources:
        output_currents[Symbol(f'I_{src.name}')] = src.i()
    for res in resistors:
        i_res = output_currents[Symbol(f'I_{res.name}')]
        res.update_current(i_res)
        output_voltages[Symbol(f'V_{res.name}')] = res.v()

    node_voltages = {}
    for node in nodes:
        if node.name == '0':
            continue
        path = list(nx.shortest_path(full_tree, '0', node.name))
        node_voltage = sy.Add()
        for j in range(1, len(path)):
            pos = path[j - 1]
            neg = path[j]
            comp = components[full_tree[pos][neg]['edge_idx']]
            if comp.pos_node.name == pos:
                node_voltage -= comp.v()
            else:
                node_voltage += comp.v()
        node_voltages[Symbol(node.name)] = node_voltage
    all_outputs = output_currents | output_voltages | node_voltages

    for v in mutual_inductors:
        L1 : Component = v[1]
        L2 : Component = v[2]
        m = sy.Symbol(v[0]) * sy.sqrt(Symbol(L1.name) * Symbol(L2.name))
        L1.add_mutual_inductance(m * L2.der())
        L2.add_mutual_inductance(m * L1.der())

    # Collect derivative equations
    deriv_eqs = [loop_eqs[i] for i in inductor_loop_rows] + [cutset_eqs[i] for i in capacitor_cutset_rows]

    # Add dependent states also as states because they may not be dependent in different switch combination
    for i_dep in dependent_currents:
        i_states = output_currents[i_dep]
        deriv_eqs.append(sy.sympify(str(i_dep - i_states).replace('I_', 'dI_')))
        states.append(i_dep)
    for u_dep  in dependent_voltages:
        u_states = output_voltages[u_dep]
        deriv_eqs.append(sy.sympify(str(u_dep - u_states).replace('V_', 'dV_')))
        states.append(u_dep)

    # Replace resistor currents with solved equations
    deriv_eqs = sy.Matrix(sy.BlockMatrix([[sy.Matrix(deriv_eqs)]]))
    for unknown, solution in solved.items():
        deriv_eqs = deriv_eqs.xreplace({unknown : solution})

    # Replace uL/iC with L*dI / C*dU
    for var in states:
        comp_name = str(var)[2:]
        comp = get_component(components, comp_name)
        if comp.name[0] == 'L':
            deriv_eqs = deriv_eqs.xreplace({comp.v() : comp.v_der()})
        elif comp.name[0] == 'C':
            deriv_eqs = deriv_eqs.xreplace({comp.i() : comp.i_der()})
        else:
            assert False, f'Invalid state component {comp_name}'

    states = [str(s) for s in states]
    states.sort()
    states = [Symbol(s) for s in states]

    states_deriv = [Symbol(f'd{str(state)}') for state in states]
    K1, Bu = sy.linear_eq_to_matrix(deriv_eqs, states_deriv)
    # Bu is moved to the right hand side so no sign inverse needed here
    A1, Bu = sy.linear_eq_to_matrix(Bu, states)
    Bu = -Bu
    inputs = []
    for c in voltage_sources + current_sources:
        if not (c.name.startswith('I_switch_')) and not (c.name.startswith('V_switch_')):
            inputs.append(Symbol(c.name))
    inputs = [str(s) for s in inputs]
    inputs.sort()
    inputs = [Symbol(s) for s in inputs]
    B1, _ = sy.linear_eq_to_matrix(Bu, inputs)

    Cx_Du = Matrix(sy.zeros(len(outputs), 1))
    for i, output in enumerate(outputs):
        Cx_Du[i] = all_outputs[output]

    states2: T.List[sy.Symbol] = []
    for state in states:
        if str(state)[0] == 'V':
            states2.append(Symbol(f'I_{str(state)[2:]}'))
        else:
            states2.append(Symbol(f'V_{str(state)[2:]}'))
    C1, C2 = sy.linear_eq_to_matrix(Cx_Du, states)
    C2, D1 = sy.linear_eq_to_matrix(-C2, states2)
    D1, _ = sy.linear_eq_to_matrix(-D1, inputs)

    # H is diagonal matrix containing L and C values
    H1 = sy.zeros(len(states))
    for i, state in enumerate(states):
        H1[i, i] = Symbol(str(state)[2:])
    K2 = C2*H1

    component_names = [c.name for c in inductors + capacitors + resistors + vv_sources + iv_sources + vi_sources + ii_sources]
    default_values = {c.name: c.default_value for c in components}
    for Lm in mutual_inductors:
        component_names.append(Lm[0])
    component_names.sort()
    ss = StateSpaceMatrices(
        component_names = component_names,
        default_values = default_values,
        states = states,
        inputs = inputs,
        outputs = outputs,
        K1 = K1,
        K2 = K2,
        A1 = A1,
        B1 = B1,
        C1 = C1,
        D1 = D1
    )
    return ss

def int_to_combination(num : int, switch_count : int) -> tuple:
    combination = [0] * switch_count
    for i in range(switch_count):
        if num & 1 << i:
            combination[i] = 1
    return tuple(combination)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('netlist', type=str)
    parser.add_argument('--dynamic', action='store_true', help='Solve only one combination with all switches open and the rest dynamically at runtime by calling this script')
    parser.add_argument('--combination', type=int, default=-1, help='Solve the circuit only for given combination and update the json. This option is only used by generated code from dynamic option.')
    args = parser.parse_args()

    filename = os.path.splitext(args.netlist)[0]
    netlist = parse_netlist(args.netlist)
    netlist, diodes = replace_diodes(netlist)
    lines_w_switches, switches, xor_switches, and_switches = lines_with_switches(netlist)

    out: dict[int, StateSpaceMatrices] = {}
    if len(lines_w_switches) == 0:
        out[0] = form_state_space_matrices(netlist)
        state_matrices_to_json.matrices_to_cpp(f'{filename}', out, switches, diodes, args.dynamic, False)
        sys.exit(0)
    else:
        if args.dynamic:
            combinations = [int_to_combination(0, len(lines_w_switches))]
        elif args.combination != -1:
            combinations = [int_to_combination(args.combination, len(lines_w_switches))]
        else:
            combinations = list(itertools.product([0, 1], repeat=len(lines_w_switches)))
        progress_bar = tqdm(total=len(combinations))
        for i, combination in enumerate(combinations):
            progress_bar.update(1)
            if is_invalid_switch_combination(combination, switches, xor_switches, and_switches):
                continue
            netlist_wo_switches = netlist.copy()
            for j, line in enumerate(lines_w_switches):
                if combination[j] == 1:
                    switch_name = line.split(' ')[0]
                    replacement = f'V_switch_{switch_name}'
                    for j, line in enumerate(netlist_wo_switches):
                        netlist_wo_switches[j] = line.replace(f'{switch_name} ', f'{replacement} ')
                else:
                    netlist_wo_switches.remove(line)
            c = [str(c) for c in list(combination)]
            c.reverse()
            combination_number = int("".join(c), 2)
            out[combination_number] = form_state_space_matrices(netlist_wo_switches)
        progress_bar.close()
    update_existing = args.combination != -1
    state_matrices_to_json.matrices_to_cpp(filename, out, switches, diodes, args.dynamic, update_existing)

if __name__ == '__main__':
    main()
