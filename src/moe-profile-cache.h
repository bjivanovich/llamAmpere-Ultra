#pragma once

#include "llama.h"
#include "moe-domain-profiles.h"
#include <cstdint>
#include <string>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

struct moe_cache_stats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> prefetches{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> total_tokens{0};

    float hit_rate() const {
        uint64_t h = hits.load();
        uint64_t m = misses.load();
        uint64_t total = h + m;
        return total > 0 ? (float)h / (float)total : 0.0f;
    }

    void reset() {
        hits.store(0);
        misses.store(0);
        prefetches.store(0);
        evictions.store(0);
        total_tokens.store(0);
    }
};

class moe_profile_cache {
public:
    static LLAMA_API moe_profile_cache & instance();

    LLAMA_API void configure(bool enabled,
                             const std::string & profile_name,
                             size_t vram_budget_mb,
                             moe_cache_policy_type policy);

    bool is_enabled() const { return enabled; }
    const std::string & get_profile_name() const { return current_profile_name; }
    size_t get_vram_budget_mb() const { return vram_budget_mb; }
    moe_cache_policy_type get_policy() const { return policy; }

    LLAMA_API bool is_expert_pinned(int32_t layer_idx, int32_t expert_idx) const;
    LLAMA_API float get_expert_priority(int32_t layer_idx, int32_t expert_idx) const;

    LLAMA_API void record_access(int32_t layer_idx, int32_t expert_idx, bool hit);
    LLAMA_API void record_prefetch(int32_t layer_idx, int32_t expert_idx);

    const moe_cache_stats & get_stats() const { return stats; }
    LLAMA_API void print_stats() const;

private:
    moe_profile_cache() = default;

    bool enabled = false;
    std::string current_profile_name = "general";
    size_t vram_budget_mb = 8192; // Default 8 GB for expert cache pool on RTX 3090
    moe_cache_policy_type policy = moe_cache_policy_type::PROFILE_PRIORITY;

    moe_domain_profile active_profile;
    moe_cache_stats stats;

    // Dynamic activation frequency map
    mutable std::mutex access_mutex;
    std::unordered_map<moe_expert_key, uint64_t, moe_expert_key_hash> dynamic_frequency;
};
