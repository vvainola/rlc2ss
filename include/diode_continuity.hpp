#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rlc2ss {

inline constexpr double DIODE_CONTINUITY_TOLERANCE = 1e-9;

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

inline bool diodeContinuityValid(DiodeContinuityMetrics const& metrics, double tolerance) {
    return metrics.discontinuity <= tolerance && metrics.complementarity_violation <= tolerance;
}

// Re-evaluate the instantaneous topology after every diode release. Opening one
// diode can redirect an algebraic capacitor/resistor current and make another
// diode reverse-biased at the same simulation time.
// evaluate_reverse_mask receives the current closed-diode mask and returns
// the subset carrying reverse current in that instantaneous topology.
template <typename EvaluateReverseMask>
uint64_t releaseReverseCurrentDiodeMask(uint64_t initial_closed_diode_mask,
                                        EvaluateReverseMask&& evaluate_reverse_mask) {
    uint64_t closed_diode_mask = initial_closed_diode_mask;
    while (closed_diode_mask != 0) {
        uint64_t reverse_current_mask = evaluate_reverse_mask(closed_diode_mask) & closed_diode_mask;
        if (reverse_current_mask == 0) {
            break;
        }
        closed_diode_mask &= ~reverse_current_mask;
    }
    return closed_diode_mask;
}

// Follow the best single-diode toggle while the residual strictly improves.
// Continuity takes priority; residuals within tolerance score as zero.
template <typename EvaluateMask>
DiodeContinuitySelection tryGreedyDiodeMask(size_t diode_count,
                                           uint64_t mask,
                                           double tolerance,
                                           EvaluateMask&& evaluate_mask) {
    if (diode_count >= 64) {
        throw std::invalid_argument("Diode continuity mask enumeration supports fewer than 64 diodes");
    }
    auto score = [tolerance](DiodeContinuityMetrics const& metrics) {
        return std::pair{
            metrics.discontinuity <= tolerance ? 0.0 : metrics.discontinuity,
            metrics.complementarity_violation <= tolerance ? 0.0 : metrics.complementarity_violation,
        };
    };
    auto metrics = evaluate_mask(mask);
    for (size_t round = 0; !diodeContinuityValid(metrics, tolerance); ++round) {
        if (round == diode_count) {
            return {};
        }
        uint64_t best_mask = mask;
        auto best_metrics = metrics;
        for (size_t diode = 0; diode < diode_count; ++diode) {
            uint64_t candidate_mask = mask ^ (uint64_t{1} << diode);
            auto candidate_metrics = evaluate_mask(candidate_mask);
            if (score(candidate_metrics) < score(best_metrics)) {
                best_mask = candidate_mask;
                best_metrics = candidate_metrics;
            }
        }
        if (best_mask == mask) {
            return {};
        }
        mask = best_mask;
        metrics = best_metrics;
    }
    return {.found = true,
            .mask = mask,
            .discontinuity = metrics.discontinuity,
            .complementarity_violation = metrics.complementarity_violation};
}

// Search every diode mask by increasing distance from the original diode state.
template <typename EvaluateMask>
DiodeContinuitySelection searchDiodeMasks(size_t diode_count,
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
            if (std::popcount(mask ^ initial_closed_diode_mask) != static_cast<int>(changes)) {
                continue;
            }

            DiodeContinuityMetrics metrics = evaluate_mask(mask);
            if (!diodeContinuityValid(metrics, tolerance)) {
                continue;
            }

            int const closed = std::popcount(mask);
            int const best_closed = std::popcount(best.mask);
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

template <typename EvaluateMask>
DiodeContinuitySelection selectDiodeContinuityMask(size_t diode_count,
                                                   uint64_t initial_closed_diode_mask,
                                                   double tolerance,
                                                   EvaluateMask&& evaluate_mask) {
    auto selection = tryGreedyDiodeMask(diode_count, initial_closed_diode_mask, tolerance, evaluate_mask);
    if (!selection.found) {
        selection = searchDiodeMasks(diode_count, initial_closed_diode_mask, tolerance, evaluate_mask);
    }
    return selection;
}

} // namespace rlc2ss
