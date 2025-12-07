// MIT License
//
// Copyright (c) 2025 vvainola
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

#pragma once

#include <cstdint>
#include <cassert>
#include <optional>

namespace rlc2ss {
inline constexpr double EPSILON = 1e-12;

class OnOffDelay {

  public:
    static const uint32_t MAX_DELAY = UINT32_MAX;
    OnOffDelay() {}
    OnOffDelay(double on_delay_time, double off_delay_time)
        : m_on_delay_time(on_delay_time), m_off_delay_time(off_delay_time) {
    }

    void operator=(bool input) {
        // Reset timer on change
        if (input != m_input) {
            m_timer = 0.0;
        }
        m_input = input;
        // Update state in case of no delay
        step(0);
    }

    operator bool() const {
        if (m_forced_output) {
            return *m_forced_output;
        }
        return m_output;
    }

    bool output() const { return m_output; }

    void forceOutput(std::optional<bool> output) { m_forced_output = output; }
    std::optional<bool> forcedOutput() const { return m_forced_output; }
    bool outputForced() const { return m_forced_output.has_value(); }

    double pendingTime() const {
        if (m_input != m_output) {
            double delay = m_input ? m_on_delay_time : m_off_delay_time;
            return delay - m_timer;
        }
        return MAX_DELAY;
    }

    bool step(double dt) {
        if (m_input != m_output) {
            m_timer += dt;
            double delay = m_input ? m_on_delay_time : m_off_delay_time;
            if (m_timer + EPSILON >= delay) {
                m_output = m_input;
                m_timer = 0.0;
            }
        }
        if (m_forced_output) {
            m_actual_output = *m_forced_output;
            return *m_forced_output;
        }
        m_actual_output = m_output;
        return m_output;
    }

    void setOnOffDelays(double on_delay_time, double off_delay_time) {
        m_on_delay_time = on_delay_time;
        m_off_delay_time = off_delay_time;
        // Update state in case existing timer is larger than new delays
        step(0);
    }

  private:
    bool m_input = false;
    bool m_output = false;
    bool m_actual_output = false;
    std::optional<bool> m_forced_output = std::nullopt;
    double m_timer = 0.0;
    double m_on_delay_time = 0.0;
    double m_off_delay_time = 0.0;
};

} // namespace rlc2ss
