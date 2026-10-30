#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <algorithm>
#include "../ggml/src/moe-python-mask.h"

enum class moe_cache_policy_type {
    PROFILE_PRIORITY = 0, // Profile experts strictly pinned, fallback on demand
    DYNAMIC_HYBRID   = 1, // Profile experts weighted with dynamic LRU/LFU tracking
    DYNAMIC_LRU      = 2  // Pure reactive LRU (similar to generic cache)
};

struct moe_expert_key {
    int32_t layer_idx;
    int32_t expert_idx;

    bool operator==(const moe_expert_key & other) const {
        return layer_idx == other.layer_idx && expert_idx == other.expert_idx;
    }
};

struct moe_expert_key_hash {
    std::size_t operator()(const moe_expert_key & k) const {
        return (std::hash<int32_t>()(k.layer_idx) ^ (std::hash<int32_t>()(k.expert_idx) << 16));
    }
};

struct moe_domain_profile {
    std::string name;
    std::string description;
    // Map of (layer, expert) -> priority weight (1.0 = critical, 0.0 = low)
    std::unordered_map<moe_expert_key, float, moe_expert_key_hash> expert_weights;
    // Set of must-pin expert keys
    std::unordered_set<moe_expert_key, moe_expert_key_hash> pinned_experts;

    float get_weight(int32_t layer, int32_t expert) const {
        moe_expert_key key{layer, expert};
        auto it = expert_weights.find(key);
        if (it != expert_weights.end()) {
            return it->second;
        }
        return 0.0f;
    }

    bool is_pinned(int32_t layer, int32_t expert) const {
        moe_expert_key key{layer, expert};
        return pinned_experts.find(key) != pinned_experts.end();
    }
};

class moe_profile_registry {
public:
    static const moe_domain_profile & get_profile(const std::string & profile_name) {
        static moe_profile_registry instance;
        auto it = instance.profiles.find(profile_name);
        if (it != instance.profiles.end()) {
            return it->second;
        }
        // Fallback to "general" if not found
        return instance.profiles["general"];
    }

    static bool has_profile(const std::string & profile_name) {
        static moe_profile_registry instance;
        return instance.profiles.find(profile_name) != instance.profiles.end();
    }

    static std::vector<std::string> get_available_profiles() {
        static moe_profile_registry instance;
        std::vector<std::string> names;
        for (const auto & kv : instance.profiles) {
            names.push_back(kv.first);
        }
        std::sort(names.begin(), names.end());
        return names;
    }

private:
    std::unordered_map<std::string, moe_domain_profile> profiles;

    moe_profile_registry() {
        init_built_in_profiles();
    }

    void init_built_in_profiles() {
        // 1. Python Profile
        {
            moe_domain_profile prof;
            prof.name = "python";
            prof.description = "Optimized for Python development, data science, script automation";
            // Complete 89.8% hot expert mapping: 48 layers x 230 hot experts = 11,040 pinned experts
            for (int l = 0; l < MOE_PYTHON_PROFILE_NUM_LAYERS; ++l) {
                for (int e = 0; e < MOE_PYTHON_PROFILE_NUM_EXPERTS; ++e) {
                    if (moe_is_python_expert_hot(l, e)) {
                        moe_expert_key k{l, e};
                        prof.expert_weights[k] = 1.0f;
                        prof.pinned_experts.insert(k);
                    }
                }
            }
            profiles["python"] = prof;
        }

        // 2. Rust Profile
        {
            moe_domain_profile prof;
            prof.name = "rust";
            prof.description = "Optimized for Rust systems programming, borrow checker patterns, type semantics";
            for (int l = 0; l < 64; ++l) {
                for (int e : {1, 2, 5, 8, 11, 16, 20, 24, 29, 30}) {
                    moe_expert_key k{l, e};
                    prof.expert_weights[k] = 1.0f;
                    if (l >= 4 && l <= 48) {
                        prof.pinned_experts.insert(k);
                    }
                }
            }
            profiles["rust"] = prof;
        }

        // 3. C++ Profile
        {
            moe_domain_profile prof;
            prof.name = "cpp";
            prof.description = "Optimized for C and C++ memory management, templates, pointers, low-level routines";
            for (int l = 0; l < 64; ++l) {
                for (int e : {0, 2, 6, 9, 13, 17, 21, 25, 27, 31}) {
                    moe_expert_key k{l, e};
                    prof.expert_weights[k] = 1.0f;
                    if (l >= 4 && l <= 48) {
                        prof.pinned_experts.insert(k);
                    }
                }
            }
            profiles["cpp"] = prof;
        }

        // 4. TypeScript / Fullstack Profile
        {
            moe_domain_profile prof;
            prof.name = "typescript";
            prof.description = "Optimized for TypeScript, JavaScript, Node.js, Web APIs, fullstack async logic";
            for (int l = 0; l < 64; ++l) {
                for (int e : {0, 1, 3, 6, 8, 12, 14, 19, 22, 27}) {
                    moe_expert_key k{l, e};
                    prof.expert_weights[k] = 1.0f;
                    if (l >= 4 && l <= 48) {
                        prof.pinned_experts.insert(k);
                    }
                }
            }
            profiles["typescript"] = prof;
        }

        // 4. Agent Profile
        {
            moe_domain_profile prof;
            prof.name = "agent";
            prof.description = "Optimized for autonomous agent loops, tool-calling, bash commands, JSON outputs";
            for (int l = 0; l < 64; ++l) {
                for (int e : {0, 3, 7, 10, 14, 19, 23, 26, 28, 30}) {
                    moe_expert_key k{l, e};
                    prof.expert_weights[k] = 1.0f;
                    if (l >= 2 && l <= 54) {
                        prof.pinned_experts.insert(k);
                    }
                }
            }
            profiles["agent"] = prof;
        }

        // 5. General Profile
        {
            moe_domain_profile prof;
            prof.name = "general";
            prof.description = "Balanced default profile across reasoning, coding, and natural language";
            for (int l = 0; l < 64; ++l) {
                for (int e = 0; e < 32; ++e) {
                    moe_expert_key k{l, e};
                    prof.expert_weights[k] = 0.5f;
                }
            }
            profiles["general"] = prof;
        }
    }
};
