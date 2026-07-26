#include <catch2/catch_test_macros.hpp>

#include "diode_continuity.hpp"

#include <cstdint>
#include <unordered_map>

namespace {

using rlc2ss::DiodeContinuityMetrics;

auto mapEvaluator(std::unordered_map<uint64_t, DiodeContinuityMetrics> const& metrics_by_mask) {
    return [&metrics_by_mask](uint64_t mask) {
        return metrics_by_mask.at(mask);
    };
}

} // namespace

TEST_CASE("Diode continuity selector can choose a multi-diode continuity path") {
    std::unordered_map<uint64_t, DiodeContinuityMetrics> metrics{
        {0b00, {.discontinuity = 10.0, .complementarity_violation = 0.0}},
        {0b01, {.discontinuity = 5.0, .complementarity_violation = 0.0}},
        {0b10, {.discontinuity = 6.0, .complementarity_violation = 0.0}},
        {0b11, {.discontinuity = 0.0, .complementarity_violation = 0.0}},
    };

    auto selection = rlc2ss::selectDiodeContinuityMask(2, 0b00, 1e-9, mapEvaluator(metrics));

    REQUIRE(selection.found);
    CHECK(selection.mask == 0b11);
}

TEST_CASE("Diode continuity selector rejects complementarity violations") {
    std::unordered_map<uint64_t, DiodeContinuityMetrics> metrics{
        {0b00, {.discontinuity = 10.0, .complementarity_violation = 0.0}},
        {0b01, {.discontinuity = 0.0, .complementarity_violation = 1e-3}},
        {0b10, {.discontinuity = 0.0, .complementarity_violation = 0.0}},
        {0b11, {.discontinuity = 0.0, .complementarity_violation = 2e-3}},
    };

    auto selection = rlc2ss::selectDiodeContinuityMask(2, 0b00, 1e-9, mapEvaluator(metrics));

    REQUIRE(selection.found);
    CHECK(selection.mask == 0b10);
}

TEST_CASE("Diode continuity selector uses deterministic tie breaking") {
    std::unordered_map<uint64_t, DiodeContinuityMetrics> metrics{
        {0b000, {.discontinuity = 10.0, .complementarity_violation = 0.0}},
        {0b001, {.discontinuity = 0.0, .complementarity_violation = 0.0}},
        {0b010, {.discontinuity = 0.0, .complementarity_violation = 0.0}},
        {0b011, {.discontinuity = 0.0, .complementarity_violation = 0.0}},
        {0b100, {.discontinuity = 0.0, .complementarity_violation = 0.0}},
        {0b101, {.discontinuity = 0.0, .complementarity_violation = 0.0}},
        {0b110, {.discontinuity = 0.0, .complementarity_violation = 0.0}},
        {0b111, {.discontinuity = 0.0, .complementarity_violation = 0.0}},
    };

    auto selection = rlc2ss::selectDiodeContinuityMask(3, 0b011, 1e-9, mapEvaluator(metrics));

    REQUIRE(selection.found);
    CHECK(selection.mask == 0b011);
}

TEST_CASE("Diode continuity selector reports no solution") {
    std::unordered_map<uint64_t, DiodeContinuityMetrics> metrics{
        {0b00, {.discontinuity = 10.0, .complementarity_violation = 0.0}},
        {0b01, {.discontinuity = 0.0, .complementarity_violation = 1e-3}},
        {0b10, {.discontinuity = 5.0, .complementarity_violation = 0.0}},
        {0b11, {.discontinuity = 0.0, .complementarity_violation = 1e-3}},
    };

    auto selection = rlc2ss::selectDiodeContinuityMask(2, 0b00, 1e-9, mapEvaluator(metrics));

    CHECK(!selection.found);
}

TEST_CASE("Diode continuity selector stops at the first valid change distance") {
    size_t evaluations = 0;
    auto evaluator = [&evaluations](uint64_t mask) {
        ++evaluations;
        REQUIRE(mask != 0b011);
        REQUIRE(mask != 0b101);
        REQUIRE(mask != 0b110);
        REQUIRE(mask != 0b111);

        if (mask == 0b001) {
            return DiodeContinuityMetrics{.discontinuity = 0.0, .complementarity_violation = 0.0};
        }
        return DiodeContinuityMetrics{.discontinuity = 10.0, .complementarity_violation = 0.0};
    };

    auto selection = rlc2ss::selectDiodeContinuityMask(3, 0b000, 1e-9, evaluator);

    REQUIRE(selection.found);
    CHECK(selection.mask == 0b001);
    CHECK(evaluations == 4);
}
