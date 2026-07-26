#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>

#include "integrator.hpp"

namespace {

using Vector = Eigen::Matrix<double, 1, 1>;
using Matrix = Eigen::Matrix<double, 1, 1>;

struct TimeDependentSystem {
    Vector dxdt(Vector const&, double t) const {
        return Vector::Constant(t);
    }
};

} // namespace

TEST_CASE("Tustin evaluates derivatives at both ends of the timestep") {
    Integrator<Vector, Matrix, Matrix> integrator;
    integrator.updateJacobian(Matrix::Zero());

    Vector const result = integrator.stepTustin(
        TimeDependentSystem{}, Vector::Zero(), 2.0, 1.0);

    CHECK(result[0] == 2.5);
}

TEST_CASE("Linear Tustin discretizes rectangular B and updates coefficients when B changes") {
    using InputMatrix = Eigen::Matrix<double, 1, 2>;
    using InputVector = Eigen::Vector<double, 2>;

    Integrator<Vector, Matrix, InputMatrix> integrator;
    integrator.enableInverseMatrixCaching(true);

    Matrix A;
    A << -1.0;
    InputMatrix B;
    B << 2.0, 3.0;
    InputVector u;
    u << 4.0, 5.0;

    integrator.updateSystem(A, B);
    Vector result = integrator.stepLinearTustin(Vector::Constant(1.0), u, 0.1);

    CHECK(result[0] == Catch::Approx((0.95 + 0.1 * 23.0) / 1.05));

    // B is part of the cache identity. Reusing A and dt with another B must
    // not reuse the previous discretized input matrix.
    B << 1.0, 0.0;
    integrator.updateSystem(A, B);
    result = integrator.stepLinearTustin(Vector::Constant(1.0), u, 0.1);

    CHECK(result[0] == Catch::Approx((0.95 + 0.1 * 4.0) / 1.05));
}

TEST_CASE("Linear backward Euler discretizes a rectangular input matrix") {
    using InputMatrix = Eigen::Matrix<double, 1, 2>;
    using InputVector = Eigen::Vector<double, 2>;

    Integrator<Vector, Matrix, InputMatrix> integrator;
    integrator.enableInverseMatrixCaching(true);

    Matrix A;
    A << -1.0;
    InputMatrix B;
    B << 2.0, 3.0;
    InputVector u;
    u << 4.0, 5.0;

    integrator.updateSystem(A, B);
    Vector const result =
        integrator.stepLinearBackwardEuler(Vector::Constant(1.0), u, 0.1);

    CHECK(result[0] == Catch::Approx((1.0 + 0.1 * 23.0) / 1.1));
}
