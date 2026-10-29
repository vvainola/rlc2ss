#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "fixtures/input_current_matrices.hpp"

TEST_CASE("Diode continuity uses the updated input-dependent inductor current") {
    double dt = GENERATE(0.0, 0.01);
    Model_input_current model;
    model.components.L1 = 1.0;
    model.components.R1 = 1.0;
    model.components.R_D1 = 1e-3;

    Model_input_current::Inputs inputs;
    inputs.I1 = 1.0;
    inputs.V_D1 = 100.0;
    model.states.I_L1 = 1.0;
    model.switches.S1 = true;
    model.step(0.0, inputs);

    // The source imposes the new inductor current before the switch opens.
    // Comparing against the old current rejects every candidate diode mask.
    inputs.I1 = 2.0;
    model.switches.S1 = false;
    REQUIRE_NOTHROW(model.step(dt, inputs));
    CHECK(model.states.I_L1 == Catch::Approx(2.0).margin(1e-12));
    CHECK(model.outputs.I_L1 == Catch::Approx(2.0).margin(1e-12));
}
