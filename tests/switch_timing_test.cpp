#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../qucs/diode_matrices.hpp"

namespace {

Model_diode::Components testComponents() {
    Model_diode::Components components;
    components.R_D1 = 1e-3;
    components.R_D2 = 1e-3;
    components.R_D3 = 1e-3;
    return components;
}

Model_diode::Inputs testInputs() {
    Model_diode::Inputs inputs;
    inputs.V1 = 1.0;
    // Keep the diodes open so this test isolates the controlled-switch delay.
    inputs.V_D1 = 1e6;
    inputs.V_D2 = 1e6;
    inputs.V_D3 = 1e6;
    return inputs;
}

void checkModelsEqual(Model_diode const& actual, Model_diode const& expected) {
    for (size_t i = 0; i < Model_diode::NUM_STATES; ++i) {
        CHECK(actual.states.data[i] == Catch::Approx(expected.states.data[i]).margin(1e-12));
    }
    for (size_t i = 0; i < Model_diode::NUM_OUTPUTS; ++i) {
        CHECK(actual.outputs.data[i] == Catch::Approx(expected.outputs.data[i]).margin(1e-12));
    }
}

} // namespace

TEST_CASE("Delayed switch uses the old topology until its event time") {
    Model_diode delayed(testComponents());
    Model_diode split(testComponents());
    Model_diode::Inputs inputs = testInputs();

    delayed.switches.S1.setOnOffDelays(0.25, 0.0);
    delayed.switches.S1 = true;

    split.step(0.25, inputs);
    split.switches.S1 = true;
    split.step(0.75, inputs);

    delayed.step(1.0, inputs);

    checkModelsEqual(delayed, split);
}

TEST_CASE("Delayed switch refreshes outputs at a timestep endpoint") {
    Model_diode delayed(testComponents());
    Model_diode split(testComponents());
    Model_diode::Inputs inputs = testInputs();

    delayed.switches.S1.setOnOffDelays(1.0, 0.0);
    delayed.switches.S1 = true;

    split.step(1.0, inputs);
    split.switches.S1 = true;
    split.step(0.0, inputs);

    delayed.step(1.0, inputs);

    checkModelsEqual(delayed, split);
}
