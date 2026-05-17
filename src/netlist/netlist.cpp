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

#include "netlist.hpp"
#include "str_helpers.h"

#include <filesystem>
#include <fstream>

namespace rlc2ss {

std::vector<std::string> replaceDiodes(std::vector<std::string> netlist_lines) {
    std::vector<std::string> processed_lines;
    for (std::string line : netlist_lines) {
        line = str::replaceAll(line, "  ", " ");
        if (line[0] == 'D') {
            std::vector<std::string> line_split = str::split(line, ' ');
            if (line_split.size() >= 3) {
                std::string diode_name = line_split[0];
                std::string pos_node = line_split[1];
                std::string neg_node = line_split[2];
                processed_lines.push_back(std::format("V_{} {} _N_{}_1 Vp;", diode_name, pos_node, diode_name));
                processed_lines.push_back(std::format("S_{} _N_{}_1 _N_{}_2", diode_name, diode_name, diode_name));
                processed_lines.push_back(std::format("R_{} _N_{}_2 {} 1e-6;Vn;I;", diode_name, diode_name, neg_node));
            }
        } else {
            processed_lines.push_back(line);
        }
    }
    return processed_lines;
}

std::vector<std::string> collectNetlistLines(const std::string& netlist) {
    std::vector<std::string> netlist_lines;
    bool control_section = false;
    for (std::string line : str::split(netlist, '\n')) {
        line = str::trim(line);
        if (line.starts_with(".control")) {
            control_section = true;
            continue;
        }
        if (line.starts_with(".endc")) {
            control_section = false;
            continue;
        }
        // Skip control section and comments
        if (control_section
            || line.empty()
            || line[0] == '.'
            || line[0] == '*'
            || line[0] == ';'
            || line[0] == '#') {
            continue;
        }
        netlist_lines.push_back(line);
    }
    netlist_lines = replaceDiodes(netlist_lines);
    return netlist_lines;
}

std::vector<std::string> extractSwitches(std::vector<std::string> const& netlist_lines) {
    std::vector<std::string> switches;
    for (std::string line : netlist_lines) {
        if (line[0] == 'S') {
            auto tokens = str::split(line, ' ');
            if (tokens.size() < 3) {
                throw std::runtime_error("Invalid switch line (too few tokens): " + line);
            }
            std::string switch_name = tokens[0];
            switches.push_back(switch_name);
        }
    }
    std::sort(switches.begin(), switches.end());
    return switches;
}

std::vector<std::string> replaceSwitches(std::vector<std::string> const& netlist_lines, int combination) {
    std::vector<std::string> switches = extractSwitches(netlist_lines);

    // Replace switch lines according to combination
    std::vector<std::string> modified_netlist_lines;
    for (std::string line : netlist_lines) {
        std::optional<std::string> line_to_add = std::nullopt;
        if (line[0] != 'S') {
            modified_netlist_lines.push_back(line);
            continue;
        }
        for (size_t i = 0; i < switches.size(); ++i) {
            bool switch_on = (combination & (1 << i)) != 0;
            std::string switch_name = switches[i];
            // The space after switch name ensures we don't match similar names
            if (line.starts_with(switch_name + " ")&& switch_on) {
                modified_netlist_lines.push_back(rlc2ss::str::replaceAll(line, switch_name, std::format("{}_{}", rlc2ss::V_DUMMY, switch_name)));
            } else if (line.starts_with(switch_name + " ") && !switch_on) {
                // Skip the switch line (open switch)
            }
        }
    }
    return modified_netlist_lines;
}

Node& getOrCreateNode(std::vector<std::unique_ptr<Node>>& nodes, const std::string& name) {
    for (auto& node : nodes) {
        if (node->name() == name) {
            return *node;
        }
    }
    nodes.emplace_back(std::make_unique<Node>(name));
    return *nodes.back();
}

Netlist parseNetlist(std::string const& netlist_str,
                     int combination) {
    std::vector<std::string> netlist_lines = collectNetlistLines(netlist_str);
    netlist_lines = rlc2ss::replaceSwitches(netlist_lines, combination);

    Netlist netlist;

    // parse netlist lines and build nodes/components
    for (std::string raw_line : netlist_lines) {
        raw_line = str::replaceAll(raw_line, "  ", " "); // normalize spaces
        auto tokens = str::split(raw_line, ' ');
        const std::string& name = tokens[0];
        char t = name[0];
        if (t == 'K' || t == 'X' || t == 'Y') {
            // Mutual inductors are handled on second pass. X and Y are mutually exclusive/include switches which are ignored
            continue;
        }
        if (tokens.size() < 3) {
            throw std::runtime_error("Invalid netlist line (too few tokens): " + raw_line);
        }
        Node& pos_node = getOrCreateNode(netlist.nodes, tokens[1]);
        Node& neg_node = getOrCreateNode(netlist.nodes, tokens[2]);
        std::string default_value_txt = "-1";
        for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
            if (it->find(';') != std::string::npos) {
                default_value_txt = *it;
                break;
            }
        }

        // Use the component NAME as the symbolic value for R, L, C, and dependent sources
        SymScalar value(0.0);
        if (name[0] == 'V' || name[0] == 'I') {
            // Voltage/current sources don't have a "value" in the component sense
            value = SymScalar(0.0);
        } else if (name[0] == 'R' || name[0] == 'L' || name[0] == 'C') {
            // Use the component name as symbolic value
            value = SymScalar(name);
        } else if (name[0] == 'E' || name[0] == 'G' || name[0] == 'F' || name[0] == 'H') {
            // Use the component name as symbolic value (gain)
            value = SymScalar(name);
        } else {
            throw std::runtime_error("Unknown component type: " + raw_line);
        }
        std::unique_ptr<Component>& component = netlist.components.emplace_back(std::make_unique<Component>(name, pos_node, neg_node, value));

        // classify
        if (t == 'V') {
            netlist.voltage_sources.push_back(component.get());
        } else if (t == 'I') {
            netlist.current_sources.push_back(component.get());
        } else if (t == 'R') {
            netlist.resistors.push_back(component.get());
        } else if (t == 'C') {
            netlist.capacitors.push_back(component.get());
            netlist.outputs.push_back(std::format("V_{}", component->name()));
        } else if (t == 'L') {
            netlist.inductors.push_back(component.get());
            netlist.outputs.push_back(std::format("I_{}", component->name()));
        } else if (t == 'E') {
            netlist.vv_sources.push_back(component.get());
            Node& pos_voltage = getOrCreateNode(netlist.nodes, tokens[3]);
            Node& neg_voltage = getOrCreateNode(netlist.nodes, tokens[4]);
            component->setSourceVoltageNodes(&pos_voltage, &neg_voltage);
        } else if (t == 'F') {
            netlist.ii_sources.push_back(component.get());
            LinearExpr controlling_current = "I_" + tokens[3];
            component->setCurrent(value * controlling_current);
        } else if (t == 'G') {
            netlist.vi_sources.push_back(component.get());
            Node& pos_voltage = getOrCreateNode(netlist.nodes, tokens[3]);
            Node& neg_voltage = getOrCreateNode(netlist.nodes, tokens[4]);
            component->setSourceVoltageNodes(&pos_voltage, &neg_voltage);
        } else if (t == 'H') {
            netlist.iv_sources.push_back(component.get());
            LinearExpr controlling_current = "I_" + tokens[3];
            component->setVoltage(value * controlling_current);
        } else {
            assert(false);
        }

        std::string comp_outputs = str::upper(default_value_txt);
        if (pos_node.name()[0] == 'N') {
            netlist.outputs.push_back(pos_node.name());
        }
        if (neg_node.name()[0] == 'N') {
            netlist.outputs.push_back(neg_node.name());
        }
        if (comp_outputs.contains("VP;")) {
            netlist.outputs.push_back(pos_node.name());
        }
        if (comp_outputs.contains("VN;")) {
            netlist.outputs.push_back(neg_node.name());
        }
        if (comp_outputs.contains("VC;")) {
            if (component->name()[0] == 'V') {
                netlist.outputs.push_back(component->name());
            } else {
                netlist.outputs.push_back(std::format("V_{}", component->name()));
            }
        }
        if (comp_outputs.contains("I;")) {
            if (component->name()[0] == 'I') {
                netlist.outputs.push_back(component->name());
            } else {
                netlist.outputs.push_back(std::format("I_{}", component->name()));
            }
        }
    }
    // parse mutual inductors after all components are created
    for (const std::string& raw_line : netlist_lines) {
        if (raw_line[0] == 'K') {
            auto tokens = str::split(raw_line, ' ');
            if (tokens.size() < 3) {
                throw std::runtime_error("Invalid mutual inductor line (too few tokens): " + raw_line);
            }
            const std::string& name1 = tokens[1];
            const std::string& name2 = tokens[2];
            // Store the coupling coefficient name (e.g. "K12") symbolically
            std::string k_name = tokens[0];
            Component* comp1 = netlist.getComponent(name1);
            Component* comp2 = netlist.getComponent(name2);
            if (comp1 == nullptr || comp2 == nullptr) {
                throw std::runtime_error("Mutual inductor references unknown component(s): " + raw_line);
            }
            netlist.mutual_inductors.push_back(std::make_tuple(k_name, comp1, comp2));
        }
    }
    // Collect and sort inputs
    for (auto& src : netlist.current_sources) {
        netlist.inputs.push_back(src->name());
    }
    for (auto& src : netlist.voltage_sources) {
        if (!src->name().starts_with(V_DUMMY)) {
            netlist.inputs.push_back(src->name());
        }
    }
    std::sort(netlist.inputs.begin(), netlist.inputs.end(), [](auto const& a, auto const& b) {
        return a < b;
    });

    // Remove duplicate outputs and sort
    std::sort(netlist.outputs.begin(), netlist.outputs.end());
    netlist.outputs.erase(std::unique(netlist.outputs.begin(), netlist.outputs.end()), netlist.outputs.end());
    if (netlist.outputs[0] == "0") {
        netlist.outputs.erase(netlist.outputs.begin());
    }

    return netlist;
}

Component* Netlist::getComponent(std::string const& component_name) const {
    return components[getComponentIndex(component_name)].get();
}

int Netlist::getComponentIndex(std::string const& component_name) const {
    for (int i = 0; i < components.size(); ++i) {
        if (components[i]->name() == component_name) {
            return i;
        }
    }
    assert(("Tried to access non-existent component", 0));
    return -1;
}

} // namespace rlc2ss
