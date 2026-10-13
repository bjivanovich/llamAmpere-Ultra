#include "moe-profile-cache.h"
#include "../ggml/src/ggml-backend-moe-cache.h"
#include <cstdio>
#include <iostream>

moe_profile_cache & moe_profile_cache::instance() {
    static moe_profile_cache inst;
    return inst;
}

void moe_profile_cache::configure(bool is_enabled,
                                  const std::string & profile_name,
                                  size_t budget_mb,
                                  moe_cache_policy_type pol) {
    enabled = is_enabled;
    current_profile_name = profile_name.empty() ? "general" : profile_name;
    vram_budget_mb = budget_mb > 0 ? budget_mb : 8192;
    policy = pol;

    active_profile = moe_profile_registry::get_profile(current_profile_name);
    stats.reset();

    int prof_id = 0;
    if (current_profile_name == "python") prof_id = 1;
    else if (current_profile_name == "rust") prof_id = 2;
    else if (current_profile_name == "cpp") prof_id = 3;
    else if (current_profile_name == "typescript") prof_id = 4;
    else if (current_profile_name == "agent") prof_id = 5;

    ggml_moe_cache_set_profile(prof_id);

    std::fprintf(stdout, "[MoE Profile Cache] Initialized: profile='%s' (id=%d), budget=%zu MB, policy=%d\n",
                 active_profile.name.c_str(), prof_id, vram_budget_mb, static_cast<int>(policy));
    std::fprintf(stdout, "[MoE Profile Cache] Pinned experts count: %zu\n",
                 active_profile.pinned_experts.size());
}

bool moe_profile_cache::is_expert_pinned(int32_t layer_idx, int32_t expert_idx) const {
    if (!enabled) {
        return false;
    }
    return active_profile.is_pinned(layer_idx, expert_idx);
}

float moe_profile_cache::get_expert_priority(int32_t layer_idx, int32_t expert_idx) const {
    if (!enabled) {
        return 0.0f;
    }

    float base_weight = active_profile.get_weight(layer_idx, expert_idx);

    if (policy == moe_cache_policy_type::PROFILE_PRIORITY) {
        return base_weight;
    }

    // For dynamic hybrid policy, blend base prior with runtime frequency
    std::lock_guard<std::mutex> lock(access_mutex);
    moe_expert_key key{layer_idx, expert_idx};
    auto it = dynamic_frequency.find(key);
    float freq_boost = 0.0f;
    if (it != dynamic_frequency.end()) {
        freq_boost = std::min(1.0f, static_cast<float>(it->second) / 100.0f);
    }

    return (0.7f * base_weight) + (0.3f * freq_boost);
}

void moe_profile_cache::record_access(int32_t layer_idx, int32_t expert_idx, bool hit) {
    if (!enabled) {
        return;
    }

    if (hit) {
        stats.hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats.misses.fetch_add(1, std::memory_order_relaxed);
    }
    stats.total_tokens.fetch_add(1, std::memory_order_relaxed);

    if (policy != moe_cache_policy_type::PROFILE_PRIORITY) {
        std::lock_guard<std::mutex> lock(access_mutex);
        moe_expert_key key{layer_idx, expert_idx};
        dynamic_frequency[key]++;
    }
}

void moe_profile_cache::record_prefetch(int32_t /*layer_idx*/, int32_t /*expert_idx*/) {
    if (!enabled) {
        return;
    }
    stats.prefetches.fetch_add(1, std::memory_order_relaxed);
}

void moe_profile_cache::print_stats() const {
    if (!enabled) {
        return;
    }

    uint64_t h = stats.hits.load(std::memory_order_relaxed);
    uint64_t m = stats.misses.load(std::memory_order_relaxed);
    uint64_t p = stats.prefetches.load(std::memory_order_relaxed);
    float rate = stats.hit_rate() * 100.0f;

    std::fprintf(stdout, "\n--- [MoE Profile Cache Summary] ---\n");
    std::fprintf(stdout, "Profile:         %s\n", current_profile_name.c_str());
    std::fprintf(stdout, "VRAM Budget:     %zu MB\n", vram_budget_mb);
    std::fprintf(stdout, "Cache Hits:      %llu\n", (unsigned long long)h);
    std::fprintf(stdout, "Cache Misses:    %llu\n", (unsigned long long)m);
    std::fprintf(stdout, "Prefetches:      %llu\n", (unsigned long long)p);
    std::fprintf(stdout, "Hit Rate:        %.2f%%\n", rate);
    std::fprintf(stdout, "-----------------------------------\n\n");
}
