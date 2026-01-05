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

#include "graph.hpp"

#include <queue>
#include <unordered_set>

namespace rlc2ss {

void Graph::addComponent(Component* comp) {
    m_components.insert(comp);
    m_node_count[comp->posNode()]++;
    m_node_count[comp->negNode()]++;
}

void Graph::removeComponent(Component* comp) {
    auto it = std::find(m_components.begin(), m_components.end(), comp);
    if (it != m_components.end()) {
        m_components.erase(it);
        m_node_count[comp->posNode()]--;
        m_node_count[comp->negNode()]--;
    }
}

std::vector<Node*> Graph::nodes() const {
    std::vector<Node*> result;
    for (const auto& [node, count] : m_node_count) {
        if (count > 0) {
            result.push_back(node);
        }
    }
    return result;
}

Node* Graph::getNode(std::string const& node_name) const {
    for (const auto& [node, count] : m_node_count) {
        if (node->name() == node_name && count > 0) {
            return node;
        }
    }
    return nullptr;
}

Component* Graph::getComponent(Node* node1, Node* node2) const {
    for (Component* comp : m_components) {
        if ((comp->posNode() == node1 && comp->negNode() == node2)
            || (comp->posNode() == node2 && comp->negNode() == node1)) {
            return comp;
        }
    }
    return nullptr;
}

bool Graph::hasPath(Node* from, Node* to) const {
    if (from == to) {
        return true;
    }
    if ((m_node_count.contains(from) && m_node_count.at(from) == 0)
        || (m_node_count.contains(to) && m_node_count.at(to) == 0)) {
        return false;
    }

    // Breadth-First Search (BFS)
    std::unordered_set<Node*> visited;
    std::queue<Node*> to_visit;
    to_visit.push(from);
    visited.insert(from);
    while (!to_visit.empty()) {
        Node* current = to_visit.front();
        to_visit.pop();
        for (Component* comp : current->connections()) {
            if (!m_components.contains(comp)) {
                continue; // Skip if component is not in the graph
            }

            Node* neighbor = nullptr;
            if (comp->posNode() == current) {
                neighbor = comp->negNode();
            } else if (comp->negNode() == current) {
                neighbor = comp->posNode();
            }
            if (neighbor && !visited.contains(neighbor)) {
                if (neighbor == to) {
                    return true;
                }
                visited.insert(neighbor);
                to_visit.push(neighbor);
            }
        }
    }
    return false;
}

std::vector<Node*> Graph::dijkstra(Node* from, Node* to) const {
    if (from == to) {
        return {};
    }

    // Dijkstra's algorithm
    std::unordered_map<Node*, Component*> prev_component;
    std::unordered_map<Node*, int> distances;
    auto cmp = [&distances](Node* left, Node* right) {
        return distances[left] > distances[right];
    };
    std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> pq(cmp);
    for (const auto& pair : m_node_count) {
        distances[pair.first] = INT_MAX;
    }
    distances[from] = 0;
    pq.push(from);
    while (!pq.empty()) {
        Node* current = pq.top();
        pq.pop();
        if (current == to) {
            break;
        }
        for (Component* comp : current->connections()) {
            // Skip if component is not in the graph
            if (!m_components.contains(comp)) {
                continue;
            }
            Node* neighbor = (comp->posNode() == current) ? comp->negNode() : comp->posNode();
            int alt = distances[current] + 1; // assuming each edge has weight 1
            if (alt < distances[neighbor]) {
                distances[neighbor] = alt;
                prev_component[neighbor] = comp;
                pq.push(neighbor);
            }
        }
    }
    std::vector<Node*> path;
    for (Node* at = to; at != from;) {
        if (prev_component.find(at) == prev_component.end()) {
            return std::vector<Node*>{}; // no path found
        }
        Component* comp = prev_component[at];
        Node* prev_node = (comp->posNode() == at) ? comp->negNode() : comp->posNode();
        path.push_back(at);
        at = prev_node;
    }
    path.push_back(from);
    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace rlc2ss
