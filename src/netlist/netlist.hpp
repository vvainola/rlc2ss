#pragma once

#include <vector>
#include <string>
#include "component.hpp"
#include <unordered_map>

namespace rlc2ss {

std::vector<std::string> collectNetlistLines(const std::string& netlist_path);
std::vector<std::string> extractSwitches(const std::vector<std::string>& netlist_lines);

struct Netlist {
    std::vector<std::unique_ptr<Node>> nodes;
    std::vector<std::unique_ptr<Component>> components;

    // Classifications
    std::vector<Component*> voltage_sources;                                       // V
    std::vector<Component*> current_sources;                                       // I
    std::vector<Component*> resistors;                                             // R
    std::vector<Component*> inductors;                                             // L
    std::vector<Component*> capacitors;                                            // C
    std::vector<Component*> vv_sources;                                            // E
    std::vector<Component*> ii_sources;                                            // F
    std::vector<Component*> vi_sources;                                            // G
    std::vector<Component*> iv_sources;                                            // H
    std::vector<std::tuple<std::string, Component*, Component*>> mutual_inductors; // K

    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    Component* getComponent(std::string const& component_name) const;
    int getComponentIndex(std::string const& component_name) const;
};

Netlist parseNetlist(std::string const& netlist_str,
                     int combination,
                     std::unordered_map<std::string, double> const& component_values);

} // namespace rlc2ss
