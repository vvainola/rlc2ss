#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace rlc2ss {

struct DiodeContinuityMetrics {
    // Max mismatch between instantaneous inductor output currents and the
    // stored inductor states for one candidate diode mask.
    double discontinuity = std::numeric_limits<double>::infinity();

    // Positive values mean the candidate violates diode complementarity:
    // either a closed diode conducts backward or an open diode is forward-biased.
    double complementarity_violation = std::numeric_limits<double>::infinity();
};

struct DiodeContinuitySelection {
    bool found = false;
    uint64_t mask = 0;
    double discontinuity = std::numeric_limits<double>::infinity();
    double complementarity_violation = std::numeric_limits<double>::infinity();
};

inline uint64_t countSetBits(uint64_t value) {
    uint64_t count = 0;
    while (value != 0) {
        count += value & uint64_t{1};
        value >>= 1;
    }
    return count;
}

inline bool diodeContinuityValid(DiodeContinuityMetrics const& metrics, double tolerance) {
    return metrics.discontinuity <= tolerance && metrics.complementarity_violation <= tolerance;
}

// Search diode masks by increasing distance from the current diode state. This
// keeps normal freewheel transitions fast: if one extra diode fixes continuity,
// only masks with zero or one changed bit are evaluated. The search is still
// complete because it continues to larger distances when no nearer mask is
// valid.
template <typename EvaluateMask>
DiodeContinuitySelection selectDiodeContinuityMask(size_t diode_count,
                                                   uint64_t initial_closed_diode_mask,
                                                   double tolerance,
                                                   EvaluateMask&& evaluate_mask) {
    if (diode_count >= 64) {
        throw std::invalid_argument("Diode continuity mask enumeration supports fewer than 64 diodes");
    }

    uint64_t const mask_count = uint64_t{1} << diode_count;
    for (uint64_t changes = 0; changes <= diode_count; ++changes) {
        DiodeContinuitySelection best;
        // Among masks with the same number of changed diode states, pick a
        // deterministic minimal solution: fewer closed diodes, then smaller
        // residual discontinuity, then lower numeric mask.
        for (uint64_t mask = 0; mask < mask_count; ++mask) {
            if (countSetBits(mask ^ initial_closed_diode_mask) != changes) {
                continue;
            }

            DiodeContinuityMetrics metrics = evaluate_mask(mask);
            if (!diodeContinuityValid(metrics, tolerance)) {
                continue;
            }

            uint64_t const closed = countSetBits(mask);
            uint64_t const best_closed = countSetBits(best.mask);
            bool const better =
                !best.found ||
                closed < best_closed ||
                (closed == best_closed && metrics.discontinuity < best.discontinuity) ||
                (closed == best_closed && metrics.discontinuity == best.discontinuity && mask < best.mask);

            if (better) {
                best.found = true;
                best.mask = mask;
                best.discontinuity = metrics.discontinuity;
                best.complementarity_violation = metrics.complementarity_violation;
            }
        }
        if (best.found) {
            return best;
        }
    }
    return {};
}

} // namespace rlc2ss
