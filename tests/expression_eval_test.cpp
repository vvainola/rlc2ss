#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include "rlc2ss.h"

TEST_CASE("Evaluate expression") {
    CHECK(rlc2ss::evaluateExpression("1 + 1") == 2);
    CHECK(rlc2ss::evaluateExpression("2^10") == 1024);
    CHECK(rlc2ss::evaluateExpression("1 + -2") == -1);
    CHECK(rlc2ss::evaluateExpression("1 + -(2)") == -1);
    CHECK(rlc2ss::evaluateExpression("-3 * 5") == -15);
    CHECK(rlc2ss::evaluateExpression("(-3 * 5)") == -15);
    CHECK(rlc2ss::evaluateExpression("(-3 * 5 / 5 * 5 + 1)") == -14);
    CHECK(rlc2ss::evaluateExpression("-1") == -1);
    CHECK(rlc2ss::evaluateExpression("-(1)") == -1);
    CHECK(rlc2ss::evaluateExpression("(-1)") == -1);
    CHECK(rlc2ss::evaluateExpression("-(-3.0)") == 3);
    CHECK(rlc2ss::evaluateExpression("-(-3.0) - (-12)") == 15);
    CHECK(rlc2ss::evaluateExpression("(-1.0*2.0)") == -2);
    CHECK(rlc2ss::evaluateExpression("(-1.0*-2.0)") == 2);
    CHECK(rlc2ss::evaluateExpression("(-1.0*-2.0 + 2 + 4 / 2 / 2 + 4)") == 9);
    CHECK(rlc2ss::evaluateExpression("(-1.0+2.0)*(3.0-12)") == -9);
    CHECK(rlc2ss::evaluateExpression("(-1.0+2.0) * (3.0 -12)") == -9);
    CHECK(rlc2ss::evaluateExpression("((-1.0 + 2.0) * (3.0 - 12)) / 2.0") == -4.5);
    CHECK(rlc2ss::evaluateExpression("-4.0 * 2.0 / (4.0 + 2.0) - (4.0 * 1.0 / (4.0 + 2.0) - 2.0 * 2.0 / (4.0 + 4.0))") == -1.5);
    CHECK(rlc2ss::evaluateExpression("-8 + (-(7+ -5)-(-3/3/3+2+2*3))*3") == -37);
    CHECK(rlc2ss::evaluateExpression("(8 - 1 + 3) * 6 - ((3 + 7) * 2)") == 40);
    CHECK(rlc2ss::evaluateExpression("(7-1)*(3+1)+1*(3-7+1)") == 21);
    CHECK(rlc2ss::evaluateExpression("((7-1)/((3+1)+1))*(3-7-1)") == -6);
    CHECK(rlc2ss::evaluateExpression("((7-1)/-((3+1)/4 + 4 + 4 / 2 / 2))*-(3-7-1)") == -5);
    CHECK(rlc2ss::evaluateExpression("2^3*3") == 24);
    CHECK(rlc2ss::evaluateExpression("1 / 2^24") == (1 / pow(2,24)));
    CHECK(rlc2ss::evaluateExpression("2^ (3+1)") == 16);
    CHECK(rlc2ss::evaluateExpression("2^ ( (3+1) * 2)") == 256);
    CHECK(rlc2ss::evaluateExpression("2^ ( (3+1) * 2 + 1)") == 512);
}

