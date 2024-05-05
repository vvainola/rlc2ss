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

#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>
#include <stack>
#include <assert.h>

constexpr double EPSILON_MIN = 1e-12;

namespace rlc2ss {

static bool isOperator(char c);
static int getPrecedence(char op);
static double applyOperator(double operand1, double operand2, char op);
static std::vector<std::string> split(const std::string& s, char delim);

std::string readFile(const std::string& filename) {
    HANDLE file_handle = CreateFileA(
        filename.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);

    if (file_handle == INVALID_HANDLE_VALUE) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return "";
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle, &file_size)) {
        std::cerr << "Error getting file size: " << filename << std::endl;
        CloseHandle(file_handle);
        return "";
    }

    HANDLE file_mapping = CreateFileMapping(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (file_mapping == nullptr) {
        std::cerr << "Error creating file mapping: " << filename << std::endl;
        CloseHandle(file_handle);
        return "";
    }

    char* file_contents = static_cast<char*>(MapViewOfFile(file_mapping, FILE_MAP_READ, 0, 0, file_size.QuadPart));
    if (file_contents == nullptr) {
        std::cerr << "Error mapping file to memory: " << filename << std::endl;
        CloseHandle(file_mapping);
        CloseHandle(file_handle);
        return "";
    }

    std::string result(file_contents, file_size.QuadPart);

    if (!UnmapViewOfFile(file_contents)) {
        std::cerr << "Error unmapping file: " << filename << std::endl;
    }

    CloseHandle(file_mapping);
    CloseHandle(file_handle);

    return result;
}

std::string replace(const std::string& original, const std::string& search, const std::string& replacement) {
    std::string result = original;
    size_t pos = 0;

    while ((pos = result.find(search, pos)) != std::string::npos) {
        result.replace(pos, search.length(), replacement);
        pos += replacement.length();
    }

    return result;
}

static bool isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/';
}

static int getPrecedence(char op) {
    if (op == '+' || op == '-') {
        return 1;
    } else if (op == '*' || op == '/') {
        return 2;
    }
    return 0; // Default precedence for non-operators
}

double applyOperator(double operand1, double operand2, char op) {
    switch (op) {
        case '+': return operand1 + operand2;
        case '-': return operand1 - operand2;
        case '*': return operand1 * operand2;
        case '/': return operand1 / operand2;
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
    std::vector<std::string> values_str = rlc2ss::split(s, ',');
    std::vector<double> values;
    values.reserve(values_str.size());
    for (std::string const& v : values_str) {
        values.push_back(rlc2ss::evaluateExpression(v));
    }
    return values;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> elems;
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::string loadTextResource(int resource_id) {
    HRSRC resource_handle = FindResource(nullptr, MAKEINTRESOURCEA(resource_id), "TEXT");
    if (resource_handle == nullptr) {
        return "";
    }
    HGLOBAL memory_handle = LoadResource(nullptr, resource_handle);
    if (memory_handle == nullptr) {
        return "";
    }

    size_t size_bytes = SizeofResource(nullptr, resource_handle);
    void* ptr = LockResource(memory_handle);

    if (ptr != nullptr) {
        return std::string(reinterpret_cast<char*>(ptr), size_bytes);
    }
    return "";
}

template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

double calcZeroCrossingTime(double y1, double y2) {
    if (sign(y1) == sign(y2)) {
        return EPSILON_MIN;
    }
    return abs(y1) / (abs(y1) + abs(y2));
}

} // namespace rlc2ss
