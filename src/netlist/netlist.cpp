#include "netlist.hpp"
#include "netlist.hpp"
#include "str_helpers.h"
#pragma warning(push, 0)
#include "symengine/expression.h"
#include "symengine/solve.h"
#include "cxxgraph/CXXGraph.hpp"
#pragma warning(pop)

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
            if (line.starts_with(switch_name) && switch_on) {
                modified_netlist_lines.push_back(rlc2ss::str::replaceAll(line, switch_name, std::format("{}_{}", rlc2ss::V_DUMMY, switch_name)));
            } else if (line.starts_with(switch_name) && !switch_on) {
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
                     int combination,
                     std::unordered_map<std::string, double> const& component_values) {
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
        double value = 0;
        if (name[0] == 'V' || name[0] == 'I') {
            default_value_txt = tokens.size() > 4 ? tokens[4] : "-1";
        } else if (name[0] == 'R' || name[0] == 'L' || name[0] == 'C') {
            default_value_txt = tokens.size() > 3 ? tokens[3] : "-1";
            if (component_values.contains(name)) {
                value = component_values.at(name);
            } else if (tokens.size() > 3) {
                std::string value_txt = str::split(tokens[3], ';')[0];
                try {
                    value = std::stod(value_txt);
                } catch (const std::exception&) {
                    throw std::runtime_error(std::format("Invalid numeric value {} for component {}", value_txt, name));
                }
            } else {
                throw std::runtime_error("Missing value for component: " + name);
            }
        } else if (name[0] == 'E' || name[0] == 'G') {
            default_value_txt = tokens.size() > 5 ? tokens[5] : "-1";
            if (component_values.contains(name)) {
                value = component_values.at(name);
            } else if (tokens.size() > 5) {
                std::string value_txt = str::split(tokens[5], ';')[0];
                try {
                    value = std::stod(value_txt);
                } catch (const std::exception&) {
                    throw std::runtime_error(std::format("Invalid numeric value {} for component {}", value_txt, name));
                }
            } else {
                throw std::runtime_error("Missing value for component: " + name);
            }
        } else if (name[0] == 'F' or name[0] == 'H') {
            default_value_txt = tokens.size() > 4 ? tokens[4] : "-1";
            if (component_values.contains(name)) {
                value = component_values.at(name);
            } else if (tokens.size() > 4) {
                std::string value_txt = str::split(tokens[4], ';')[0];
                try {
                    value = std::stod(value_txt);
                } catch (const std::exception&) {
                    throw std::runtime_error(std::format("Invalid numeric value {} for component {}", value_txt, name));
                }
            } else {
                throw std::runtime_error("Missing value for component: " + name);
            }
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
            SymEngine::Expression controlling_current = "I_" + tokens[3];
            std::string gain = SYMBOLIC ? name : std::to_string(value);
            component->setCurrent(gain * controlling_current);
        } else if (t == 'G') {
            netlist.vi_sources.push_back(component.get());
            Node& pos_voltage = getOrCreateNode(netlist.nodes, tokens[3]);
            Node& neg_voltage = getOrCreateNode(netlist.nodes, tokens[4]);
            component->setSourceVoltageNodes(&pos_voltage, &neg_voltage);
        } else if (t == 'H') {
            netlist.iv_sources.push_back(component.get());
            SymEngine::Expression controlling_current = "I_" + tokens[3];
            std::string gain = SYMBOLIC ? name : std::to_string(value);
            component->setVoltage(gain * controlling_current);
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
            std::string value;
            if (component_values.contains(tokens[0])) {
                value = std::to_string(component_values.at(tokens[0]));
            } else {
                value = SYMBOLIC ? tokens[0] : tokens[3];
            }
            Component* comp1 = netlist.getComponent(name1);
            Component* comp2 = netlist.getComponent(name2);
            if (comp1 == nullptr || comp2 == nullptr) {
                throw std::runtime_error("Mutual inductor references unknown component(s): " + raw_line);
            }
            netlist.mutual_inductors.push_back(std::make_tuple(value, comp1, comp2));
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
