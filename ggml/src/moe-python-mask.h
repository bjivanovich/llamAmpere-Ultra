#pragma once
#include <cstdint>

#define MOE_PYTHON_PROFILE_NUM_LAYERS 48
#define MOE_PYTHON_PROFILE_NUM_EXPERTS 256
#define MOE_PYTHON_PROFILE_HOT_EXPERTS_PER_LAYER 230
#define MOE_PYTHON_PROFILE_COLD_EXPERTS_PER_LAYER 26

static const uint64_t g_python_expert_mask[48][4] = {
    /* layer  0 */ { 0xfffffbffffeffbfdULL, 0xf7febfdf77fffdefULL, 0xffffff9fffefffffULL, 0xcdbdf76dffffbfdfULL },
    /* layer  1 */ { 0xefff3fffffd7dfffULL, 0xffffffff7ffbe79fULL, 0xdfffdfddfd7ddfffULL, 0xebfddfff7fffffefULL },
    /* layer  2 */ { 0xffffedf7e7ffffcfULL, 0xfffeeffdbfeebfffULL, 0xffffbdbbffbfbfffULL, 0xffffe6ffef5fffffULL },
    /* layer  3 */ { 0xf7ffddf7fff7ffffULL, 0xfe9ffdf7dffcddfeULL, 0xfffdfffbfff7feffULL, 0xffeffbfffbeff7feULL },
    /* layer  4 */ { 0xbf7ffdffffffebffULL, 0xff7f7ffeaffeddffULL, 0xbeffbff77ffdf7ffULL, 0xf3ffbfffffbbffefULL },
    /* layer  5 */ { 0xfffffffffff7ffffULL, 0xfefffaeffdfdf7fbULL, 0xf7f7b6dfb27ff7ffULL, 0xeeff7ffffff7efffULL },
    /* layer  6 */ { 0xffffbbff9ff9ffffULL, 0xfdfbebfffdffffefULL, 0xffffddfd9ffffbffULL, 0xddef7f7bddffffffULL },
    /* layer  7 */ { 0x7e7cfefdbf9fffffULL, 0xfffffdeffffefdffULL, 0x9df7fffbfffffeffULL, 0xfff7ffdefbfefffdULL },
    /* layer  8 */ { 0x9ffef7fffffa7fffULL, 0xf7fff7fffbffff7fULL, 0xbffbfffe79ff7edeULL, 0xffffbfff76fffbffULL },
    /* layer  9 */ { 0xf3feffffffdff77fULL, 0xeb7ffffffffffffeULL, 0xb37f7f7ffbffffffULL, 0x37ffdf77bfbf7fffULL },
    /* layer 10 */ { 0xfd7ffffdfdffffffULL, 0xbff7bfdffeff6fffULL, 0xff7f6bfff7fbbffbULL, 0xfebfbfdf7f7fff7fULL },
    /* layer 11 */ { 0xff97ffefffebffffULL, 0xfffffbcfffffdfffULL, 0xdf3dffffbfffff5fULL, 0xbed5fffff7fff6bfULL },
    /* layer 12 */ { 0xfffdfdfeffffbfdbULL, 0xdfff73fffefbf6efULL, 0xfbfdfffbfdffffffULL, 0xffffd7b3ff7ffdffULL },
    /* layer 13 */ { 0xdfce7ff7fbffffffULL, 0xf7fbfff7fbff7fffULL, 0xd7ffeffe3bfddfffULL, 0xfbeff9fffdffffffULL },
    /* layer 14 */ { 0xffffffdfbfcfefffULL, 0xd7bc77ffffdfeeffULL, 0xf6fefffbffbffbfdULL, 0xffff7ffffefffaffULL },
    /* layer 15 */ { 0xef1ffffffffffffeULL, 0xefffeaefffffffd7ULL, 0xfbcffffffdfffbffULL, 0xef5f7dff7ffcbfffULL },
    /* layer 16 */ { 0xfeffeffdee7e6fffULL, 0x7fffeffff7fbbf7fULL, 0x7b9fffffdffffffdULL, 0xeffdffffff1fffffULL },
    /* layer 17 */ { 0x3f77fffffff7fefeULL, 0xffffff7f7ffffbafULL, 0xbeffbff7ffffffffULL, 0xbf7fdacf7f7bffffULL },
    /* layer 18 */ { 0xfbfff7dfffffffefULL, 0x7ff79fbfdffffdefULL, 0xffeff3dfffbeefffULL, 0x6fbdfffefbbfffffULL },
    /* layer 19 */ { 0xffffffffffffffffULL, 0xffe4ffffbbffffffULL, 0xf7df37f7ffffffcbULL, 0xf2dfffbde5ffdbffULL },
    /* layer 20 */ { 0xfffff5fff5ff6fffULL, 0xffffffffffdeffffULL, 0xffffefbf3effffffULL, 0xf9fffbd7775b7befULL },
    /* layer 21 */ { 0x7fffffadffeffffaULL, 0xbcfffff7bfebfffdULL, 0xf5dffdffffbd7bffULL, 0xfdfffffeff7fffffULL },
    /* layer 22 */ { 0xfffbfffb2fedf7ffULL, 0xf7fffffffeffffffULL, 0x7ffdfbffffffffffULL, 0xdfef2feff6d7f7beULL },
    /* layer 23 */ { 0xfff6ff7ff7ff7f7fULL, 0xdffcfebbbbf7ffffULL, 0xeffbff7fbffffffdULL, 0xffaeffeffeffdfffULL },
    /* layer 24 */ { 0xf7fffbffffffff7fULL, 0xdeeffffefdffffefULL, 0xfeff7ffbdffcfefeULL, 0xefff7ebdbfbffefeULL },
    /* layer 25 */ { 0xfbfdfbffefffffdfULL, 0xdfdefffbfbfffbffULL, 0xfefffffdff3fffffULL, 0xb7ffdbc7eff77ffbULL },
    /* layer 26 */ { 0xfff7fdfffffff7ffULL, 0xfed5ff79fffffdfdULL, 0xedfefff7fffb7fefULL, 0xfffffbbffbd7dffdULL },
    /* layer 27 */ { 0xbffbfffbffffffefULL, 0xfdffff97ffff7fffULL, 0x4ffbffff7fffbedfULL, 0xfddaff7fddffd7ffULL },
    /* layer 28 */ { 0xffffff7f7febffffULL, 0xfff3dfefffffefdeULL, 0xe7ffb6fff5feffffULL, 0xfffddf7ffdf6ffefULL },
    /* layer 29 */ { 0xfffffefcdfffffd7ULL, 0xfffffdfffffff7ffULL, 0x97ffffef5f57f7feULL, 0xff7baffffdefffdfULL },
    /* layer 30 */ { 0xfffdfbf7ffffffefULL, 0xfff9df7effffc9fbULL, 0xfbffdeffbefffff7ULL, 0xf7bffdfffef9ffffULL },
    /* layer 31 */ { 0xfb67f7ffffffffffULL, 0xf6fefefdfffbfff3ULL, 0xfebffffdfffffefeULL, 0xff7fbfff7ffb79bfULL },
    /* layer 32 */ { 0xfffb7ff77e7d3fffULL, 0xfffbdfdefffbdbffULL, 0xffff7fbedfffdeffULL, 0xdff7fffff7fffffdULL },
    /* layer 33 */ { 0xffbafb7fffbffafeULL, 0xffffffdffffffbefULL, 0x7fffff6fffefffffULL, 0xb9fff67eff7fffebULL },
    /* layer 34 */ { 0xffffffdffff5ffdeULL, 0xfffffbfd7fffffffULL, 0xfffff69ff5fffe9fULL, 0xfde7fdfbfefef9ffULL },
    /* layer 35 */ { 0xffff7ffffffeedffULL, 0xffbaffffdfffffffULL, 0xef7edf7fffffffffULL, 0x9eb7bfef7e3fffbdULL },
    /* layer 36 */ { 0xeffff9fbffffefffULL, 0xfffffffffff77fffULL, 0xf7ff4f7aeebfbfffULL, 0xf9fbf7df5effffffULL },
    /* layer 37 */ { 0xdffffff5fdfffffeULL, 0xfeeb3fffb7ffefffULL, 0xde6ffff7fffeffd7ULL, 0xeffff5ffeffffff7ULL },
    /* layer 38 */ { 0xeffffff7afafbfffULL, 0xfdffffdffdefffffULL, 0x7ffbfbffffffffffULL, 0xbfbb7f7f7fb7f1f7ULL },
    /* layer 39 */ { 0xdfdbffffe6f77efdULL, 0xedeefbfbbfdfffffULL, 0xfffffff7fffffbffULL, 0xfff2efffffffdffeULL },
    /* layer 40 */ { 0xffdffbf71eff6fb7ULL, 0xbfbffffffffffffeULL, 0xffbfffffddb7ffffULL, 0xff9fffdfbbf77fffULL },
    /* layer 41 */ { 0xfbedffedfbffbff7ULL, 0x7ffffbffffffbffdULL, 0xfffffbffa79bffffULL, 0xf7eee7fffeffffefULL },
    /* layer 42 */ { 0xfdbbfeffff7efff5ULL, 0xffff7ffdebffffdfULL, 0xfdfefdffdffafdfeULL, 0xfffffbffffdcfff7ULL },
    /* layer 43 */ { 0xff7fffd77ff7bf7fULL, 0xffffeefcffdfffffULL, 0xfffefbfffffff3dfULL, 0xffdedefffbbfdf7eULL },
    /* layer 44 */ { 0xffb6e7fd7ffffffbULL, 0xeff9fafffe7fffffULL, 0xdfffffedfffffffbULL, 0xbfbfeff7ffbffbdfULL },
    /* layer 45 */ { 0xefffbfbffffffffeULL, 0xfef7ff3fffefdfffULL, 0xfff9de9ff5ffda7fULL, 0xbf7ffffffffff7feULL },
    /* layer 46 */ { 0xebfdffffffdbdfffULL, 0xffdeffdfffffffbbULL, 0x65ffffffefffdd7fULL, 0xffbfffddd7f7f7ffULL },
    /* layer 47 */ { 0xf7bffff7fbffffffULL, 0x9e7f3bffdfffffdfULL, 0xff3ffff7fffbffffULL, 0xbdee7ffffdfe7ffdULL },
};

static inline bool moe_is_python_expert_hot(int32_t layer_idx, int32_t expert_idx) {
    if (layer_idx >= 0 && layer_idx < 48 && expert_idx >= 0 && expert_idx < 256) {
        return (g_python_expert_mask[layer_idx][expert_idx / 64] & (1ULL << (expert_idx % 64))) != 0;
    }
    if (expert_idx >= 0 && expert_idx < 256) {
        int hot_count = 0;
        for (int l = 0; l < 48; ++l) {
            if ((g_python_expert_mask[l][expert_idx / 64] & (1ULL << (expert_idx % 64))) != 0) {
                hot_count++;
            }
        }
        return hot_count >= 24;
    }
    return false;
}
