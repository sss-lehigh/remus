// [mfs] This is only used by the atree.  Do we still need it?
#if defined(__clang__)
  #define REMUS_PACKED __attribute__((packed))
#else
  #define REMUS_PACKED
#endif
