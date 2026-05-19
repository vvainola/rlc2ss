// MIT License
//
// Copyright (c) 2024 vvainola
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

#include <iostream>
#include <fstream>
#include <sstream>
#include <stack>
#include <assert.h>
#include <format>
#include <cmath>
#include <exception>

#ifdef _WIN32
#include <windows.h>
#endif

constexpr double EPSILON_MIN = 1e-12;

namespace rlc2ss {

static bool isOperator(char c);
static int getPrecedence(char op);
static double applyOperator(double operand1, double operand2, char op);

std::string replace(const std::string& original, const std::string& search, const std::string& replacement) {
    return str::replaceAll(original, search, replacement);
}

static bool isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '^';
}

static int getPrecedence(char op) {
    if (op == '+' || op == '-') {
        return 1;
    } else if (op == '*' || op == '/') {
        return 2;
    } else if (op == '^') {
        return 3;
    }
    return 0; // Default precedence for non-operators
}

double applyOperator(double operand1, double operand2, char op) {
    switch (op) {
        case '+': return operand1 + operand2;
        case '-': return operand1 - operand2;
        case '*': return operand1 * operand2;
        case '/': return operand1 / operand2;
        case '^': return pow(operand1, operand2);
        default:
            std::cerr << "Invalid operator: " << op << std::endl;
            return 0.0; // Handle invalid operators gracefully
    }
}

double evaluateExpression(std::istringstream& iss) {
    std::stack<double> operand_stack;
    std::stack<char> operator_stack;
    std::stack<char> full_stack;

    auto evaluateOperatorStack = [&]() {
        char top_operator = operator_stack.top();
        operator_stack.pop();

        assert(operand_stack.size() >= 2);
        double operand2 = operand_stack.top();
        operand_stack.pop();
        double operand1 = operand_stack.top();
        operand_stack.pop();

        double result = applyOperator(operand1, operand2, top_operator);
        operand_stack.push(result);
    };

    char current_char;
    while (iss.get(current_char)) {
        // Digit
        // Unary operator
        // Unary operator preceded by operator e.g. "1 + -2" or "1 / -(2)"
        if (isdigit(current_char)
            || (current_char == '-' && operand_stack.empty() && (isdigit(iss.peek()) || iss.peek() == '('))
            || (current_char == '-' && !full_stack.empty() && isOperator(full_stack.top()) && (isdigit(iss.peek()) || iss.peek() == '('))) {
            // Parse a number
            double operand;
            if (iss.peek() == '(') {
                iss.get(); // Remove opening parenthesis
                operand = -evaluateExpression(iss);
            } else {
                iss.putback(current_char);
                iss >> operand;
            }
            operand_stack.push(operand);
            // Don't care about the value in full stack but it has to be distinguishable from operator
            full_stack.push('0');
        }
        // sqrt
        else if (current_char == 's') {
            assert(iss.get() == 'q');
            assert(iss.get() == 'r');
            assert(iss.get() == 't');
            assert(iss.get() == '(');
            double operand = evaluateExpression(iss);
            operand_stack.push(sqrt(operand));
            full_stack.push('0');
        } else if (isOperator(current_char)) {
            // Token is an operator
            char current_operator = current_char;

            while (!operator_stack.empty() && getPrecedence(operator_stack.top()) >= getPrecedence(current_operator)) {
                // Apply higher or equal precedence operators on top of the operator stack
                evaluateOperatorStack();
            }

            // Push the current operator onto the stack
            operator_stack.push(current_operator);
            full_stack.push(current_operator);
        } else if (current_char == '(') {
            // Token is an opening parenthesis, evaluate the expression inside the parenthesis
            double result = evaluateExpression(iss);
            operand_stack.push(result);
            // Don't care about the value in full stack but it has to be distinguishable from operator
            full_stack.push('0');
        } else if (current_char == ')') {
            // Token is a closing parenthesis, evaluate the expression
            while (!operator_stack.empty()) {
                evaluateOperatorStack();
            }
            assert(operand_stack.size() == 1);
            return operand_stack.top();
        } else {
            std::cerr << "Invalid character: " << current_char << std::endl;
            return 0.0; // Handle invalid characters gracefully
        }
    }

    // Process the remaining operators in the stack
    while (!operator_stack.empty()) {
        evaluateOperatorStack();
    }

    // The final result is on top of the operand stack
    if (operand_stack.size() == 1) {
        return operand_stack.top();
    } else {
        std::cerr << "Invalid expression: Too many operands" << std::endl;
        return 0.0;
    }
}

double evaluateExpression(std::string expression) {
    // Remove whitespace
    expression.erase(std::remove_if(expression.begin(), expression.end(), isspace), expression.end());
    std::istringstream iss(expression);
    return evaluateExpression(iss);
}

std::vector<double> getCommaDelimitedValues(std::string const s) {
    std::vector<std::string_view> values_str = str::splitSv(s, ',');
    std::vector<double> values;
    values.reserve(values_str.size());
    for (std::string_view v : values_str) {
        values.push_back(evaluateExpression(std::string(v)));
    }
    return values;
}

template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

double calcZeroCrossingTime(double y1, double y2) {
    if (sign(y1) == sign(y2)) {
        return EPSILON_MIN;
    }
    return fabs(y1) / (fabs(y1) + fabs(y2));
}

namespace {

// Memoised evaluation of an AST node. Each unique node-by-pointer is visited
// at most once per call; shared subtrees in the DAG (e.g. the `factor` reused
// across a Gaussian-elim row) are evaluated exactly once.
double evaluateNode(ExprNode const& n,
                    std::unordered_map<std::string, double> const& vars,
                    std::unordered_map<ExprNode const*, double>& cache) {
    auto it = cache.find(&n);
    if (it != cache.end()) {
        return it->second;
    }
    double v = 0.0;
    switch (n.op) {
        case ExprNode::Op::Var: {
            auto vit = vars.find(n.name);
            assert(vit != vars.end() && "Missing symbolic variable value");
            v = vit->second;
            break;
        }
        case ExprNode::Op::Const: v = n.value; break;
        case ExprNode::Op::Add:   v = evaluateNode(*n.lhs, vars, cache) + evaluateNode(*n.rhs, vars, cache); break;
        case ExprNode::Op::Sub:   v = evaluateNode(*n.lhs, vars, cache) - evaluateNode(*n.rhs, vars, cache); break;
        case ExprNode::Op::Neg:   v = -evaluateNode(*n.lhs, vars, cache); break;
        case ExprNode::Op::Mul:   v = evaluateNode(*n.lhs, vars, cache) * evaluateNode(*n.rhs, vars, cache); break;
        case ExprNode::Op::Div:   v = evaluateNode(*n.lhs, vars, cache) / evaluateNode(*n.rhs, vars, cache); break;
        case ExprNode::Op::Sqrt:  v = std::sqrt(evaluateNode(*n.lhs, vars, cache)); break;
    }
    cache[&n] = v;
    return v;
}

} // namespace

Eigen::MatrixXd evaluate(SymbolicMatrix const& m,
                         std::unordered_map<std::string, double> const& values) {
    Eigen::MatrixXd out(m.rows(), m.cols());
    std::unordered_map<ExprNode const*, double> cache;
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            SymScalar const& s = m(i, j);
            if (s.isNumeric()) {
                out(i, j) = s.numeric();
            } else {
                auto tree = s.tree();
                assert(tree);
                out(i, j) = evaluateNode(*tree, values, cache);
            }
        }
    }
    return out;
}

} // namespace rlc2ss
