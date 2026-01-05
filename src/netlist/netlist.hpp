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
