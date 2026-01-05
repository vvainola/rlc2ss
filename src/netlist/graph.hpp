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

#include "component.hpp"

#include <vector>
#include <set>
#include <unordered_map>

namespace rlc2ss {

class Graph {
  public:
    Graph() {}

    void addComponent(Component* comp);
    void removeComponent(Component* comp);
    std::vector<Node*> nodes() const;
    Node* getNode(std::string const& node_name) const;
    Component* getComponent(Node* node1, Node* node2) const;

    bool hasPath(Node* from, Node* to) const;
    std::vector<Node*> dijkstra(Node* from, Node* to) const;

  private:
    std::unordered_map<Node*, int> m_node_count;
    std::set<Component*> m_components;
};

} // namespace rlc2ss
