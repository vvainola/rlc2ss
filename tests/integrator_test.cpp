#include <catch2/catch_test_macros.hpp>

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
    Integrator<Vector, Matrix> integrator;
    integrator.updateJacobian(Matrix::Zero());

    Vector const result = integrator.stepTustin(
        TimeDependentSystem{}, Vector::Zero(), 2.0, 1.0);

    CHECK(result[0] == 2.5);
}
