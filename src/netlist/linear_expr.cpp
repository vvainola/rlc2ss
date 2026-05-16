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

#include "linear_expr.hpp"

#include <set>

namespace rlc2ss {

// Multiply matrix with vector
std::vector<LinearExpr> operator*(const Eigen::MatrixXd& mat, const std::vector<LinearExpr>& vec) {
    assert(mat.cols() == vec.size());
    std::vector<LinearExpr> result(mat.rows());
    for (int i = 0; i < mat.rows(); ++i) {
        LinearExpr row_expr;
        for (int j = 0; j < mat.cols(); ++j) {
            row_expr += vec[j] * mat(i, j);
        }
        result[i] = row_expr;
    }
    return result;
}

SymbolicSystem linearEqsToMatrix(const std::vector<LinearExpr>& eqns, const std::vector<std::string>& unknowns) {
    int rows = eqns.size();
    int cols = unknowns.size();

    SymbolicSystem sys;
    sys.A = SymbolicMatrix::Zero(rows, cols);
    sys.b.resize(rows);

    for (int i = 0; i < rows; ++i) {
        // Handle the constant part of the equation (move to RHS)
        sys.b[i].constant = -eqns[i].constant;

        for (auto const& [name, coeff] : eqns[i].terms) {
            bool found = false;
            // Check if this term belongs in A or b
            for (int j = 0; j < cols; ++j) {
                if (name == unknowns[j]) {
                    sys.A(i, j) = coeff;
                    found = true;
                    break;
                }
            }

            // If not an unknown, move it to the b matrix (sign flip)
            if (!found) {
                sys.b[i].terms[name] = -coeff;
            }
        }
        sys.b[i].updateStr();
    }
    return sys;
}

// Solves Ax = b using Gaussian elimination with symbolic scalars
std::vector<LinearExpr> solveLinearSystem(SymbolicMatrix A, std::vector<LinearExpr> b, const std::vector<std::string>& unknowns) {
    int n = unknowns.size();

    // Forward Elimination
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int pivot = i;
        while (pivot < n && A(pivot, i).isZero())
            pivot++;

        if (pivot == n) {
            continue;
        }

        // Swap rows in A and b
        if (pivot != i) {
            for (int k = 0; k < n; ++k) {
                SymScalar temp = A(i, k);
                A(i, k) = A(pivot, k);
                A(pivot, k) = temp;
            }
            LinearExpr temp_b = b[i];
            b[i] = b[pivot];
            b[pivot] = temp_b;
        }

        // Eliminate
        for (int j = i + 1; j < n; ++j) {
            if (A(j, i).isZero()) {
                continue;
            }
            SymScalar factor = A(j, i) / A(i, i);
            for (int k = i; k < n; ++k) {
                A(j, k) -= factor * A(i, k);
            }
            b[j] = b[j] - factor * b[i];
        }
    }

    // Back Substitution
    std::vector<LinearExpr> x(n);
    for (int i = n - 1; i >= 0; --i) {
        LinearExpr sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum = sum - A(i, j) * x[j];
        }
        x[i] = sum / A(i, i);
    }
    return x;
}

} // namespace rlc2ss
