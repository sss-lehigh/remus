#include <cstdio>

#include <rome/hds/allocator/allocator.h>
#include <rome/hds/unordered_map/kv_linked_list/lock_linked_list.h>
#include <rome/hds/unordered_map/kv_linked_list/locked_nodes/reg_cached_nodes.h>
#include <rome/hds/threadgroup/threadgroup.h>
#include <set>

HDS_HOST_DEVICE void error() {
#if defined(__CUDA_ARCH__)
__trap();
#else
exit(1);
#endif
}

#define ASSERT(x, y) if(!(x)) { printf("%s did not evaluate to true for i = %d\n", #x, (y)); error(); }

using namespace rome::hds::kv_linked_list;
using namespace rome::hds;

int main() {
  return 1;
}

