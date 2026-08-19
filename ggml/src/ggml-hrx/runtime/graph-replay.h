#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace ggml::hrx {

enum class HrxGraphReplayEvent {
    Disabled,
    Ineligible,
    MissBuild,
    Hit,
    RebuildTransient,
    BuildFailed,
    LaunchFailed,
};

inline constexpr bool hrx_graph_replay_enabled_by_default() {
#if defined(_WIN32)
    // Windows replay has unresolved lifetime and binding behavior that can cause GPU timeouts.
    // Use direct execution by default until replay is fixed.
    return false;
#else
    return true;
#endif
}

inline bool hrx_graph_replay_enabled_from_environment() {
#if defined(_WIN32)
    const char * value = std::getenv("GGML_HRX_ENABLE_GRAPH_REPLAY");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
#else
    return hrx_graph_replay_enabled_by_default();
#endif
}

inline bool hrx_graph_replay_should_fallback(HrxGraphReplayEvent event) {
    return event == HrxGraphReplayEvent::Disabled || event == HrxGraphReplayEvent::Ineligible ||
           event == HrxGraphReplayEvent::BuildFailed;
}

inline const char * hrx_graph_replay_event_name(HrxGraphReplayEvent event) {
    switch (event) {
        case HrxGraphReplayEvent::Disabled:
            return "disabled";
        case HrxGraphReplayEvent::Ineligible:
            return "ineligible";
        case HrxGraphReplayEvent::MissBuild:
            return "miss_build";
        case HrxGraphReplayEvent::Hit:
            return "hit";
        case HrxGraphReplayEvent::RebuildTransient:
            return "rebuild_transient";
        case HrxGraphReplayEvent::BuildFailed:
            return "build_failed";
        case HrxGraphReplayEvent::LaunchFailed:
            return "launch_failed";
    }
    return "unknown";
}

inline uint64_t hrx_graph_replay_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace ggml::hrx
