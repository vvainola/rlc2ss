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

std::unordered_set<Node*> reachableNodes(Node* start, std::unordered_set<Component*> components) {
    // Breadth-First Search (BFS)
    std::unordered_set<Node*> visited;
    std::queue<Node*> to_visit;
    to_visit.push(start);
    visited.insert(start);
    while (!to_visit.empty()) {
        Node* current = to_visit.front();
        to_visit.pop();
        for (Component* comp : current->connections()) {
            if (!components.contains(comp)) {
                continue; // Skip if component is not in the graph
            }

            Node* neighbor = nullptr;
            if (comp->posNode() == current) {
                neighbor = comp->negNode();
            } else if (comp->negNode() == current) {
                neighbor = comp->posNode();
            }
            if (neighbor && !visited.contains(neighbor)) {
                visited.insert(neighbor);
                to_visit.push(neighbor);
            }
        }
    }
    return visited;
}

void Graph::addComponent(Component* comp) {
    m_components.insert(comp);

    // Find islands containing the component's nodes
    int pos_idx = -1;
    int neg_idx = -1;
    for (size_t i = 0; i < m_islands.size(); ++i) {
        if (m_islands[i].contains(comp->posNode())) {
            pos_idx = static_cast<int>(i);
        }
        if (m_islands[i].contains(comp->negNode())) {
            neg_idx = static_cast<int>(i);
        }
        // Early exit if both found
        if (pos_idx != -1 && neg_idx != -1) {
            break;
        }
    }

    if (pos_idx != -1 && neg_idx != -1 && pos_idx != neg_idx) {
        // Merge islands; erase the higher index to keep indices valid
        if (pos_idx < neg_idx) {
            m_islands[pos_idx].insert(m_islands[neg_idx].begin(), m_islands[neg_idx].end());
            m_islands.erase(m_islands.begin() + neg_idx);
        } else {
            m_islands[neg_idx].insert(m_islands[pos_idx].begin(), m_islands[pos_idx].end());
            m_islands.erase(m_islands.begin() + pos_idx);
        }
        return;
    }

    // One node found - add the other node to that island
    if (pos_idx != -1) {
        m_islands[pos_idx].insert(comp->negNode());
        return;
    }

    // One node found - add the other node to that island
    if (neg_idx != -1) {
        m_islands[neg_idx].insert(comp->posNode());
        return;
    }

    // Neither node found - create a new island
    m_islands.push_back({comp->posNode(), comp->negNode()});
}

void Graph::removeComponent(Component* comp) {
    auto it = std::find(m_components.begin(), m_components.end(), comp);
    if (it != m_components.end()) {
        m_components.erase(it);
    }
    for (auto& island : m_islands) {
        // If the island contains either node, we need to recompute reachable nodes
        if (island.contains(comp->posNode())) {
            island.clear();
            // Recompute nodes reachable from posNode
            std::unordered_set<Node*> reachable_pos = reachableNodes(comp->posNode(), m_components);
            island.insert(reachable_pos.begin(), reachable_pos.end());
            // Create new island for nodes reachable from negNode if not already in the same island
            if (!island.contains(comp->negNode())) {
                std::unordered_set<Node*> reachable_neg = reachableNodes(comp->negNode(), m_components);
                m_islands.push_back(reachable_neg);
            }
            return;
        }
    }
}

std::vector<Node*> Graph::nodes() const {
    std::vector<Node*> result;
    for (const auto& component : m_components) {
        result.push_back(component->posNode());
        result.push_back(component->negNode());
    }
    return result;
}

Node* Graph::getNode(std::string const& node_name) const {
    for (const auto& component : m_components) {
        if (component->posNode()->name() == node_name) {
            return component->posNode();
        }
        if (component->negNode()->name() == node_name) {
            return component->negNode();
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
    // Check if both nodes are in the same island
    for (const auto& island : m_islands) {
        if (island.contains(from) && island.contains(to)) {
            return true;
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
    for (const auto& component : m_components) {
        distances[component->posNode()] = INT_MAX;
        distances[component->negNode()] = INT_MAX;
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
