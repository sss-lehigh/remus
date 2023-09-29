#if defined(__clang__)
#define ROME_PACKED __attribute__((packed))
#else
#define ROME_PACKED
#endif