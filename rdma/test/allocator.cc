#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>

#include <protos/workloaddriver.pb.h>
#include <remus/logging/logging.h>
#include <remus/rdma/memory_pool.h>
#include <remus/rdma/rdma.h>

using namespace remus::rdma;

int exit_code = 0;
#define REMUS_TEST(assert_true, ...) \
    if (!(assert_true)) { REMUS_ERROR(__VA_ARGS__); exit_code = 1; }

class FakeObject {
    long start;
    long end;
};

class alignas(64) FakeObject64 {
    long start;
    long end;
};

class alignas(128) FakeObject128 {
    long start;
    long end;
};

class alignas(256) FakeObject256 {
    long start;
    long end;
};

template <typename t>
bool aligned_to(t* p, uint64_t value){
    // value should be power of 2
    return ((uint64_t) p & ~(1 - value)) == 0;
}

template <typename t1, typename t2>
inline uint64_t ptr_cmp(t1* p1, t2* p2){
    return (uint64_t) p1 - (uint64_t) p2;
}

int main() {
    int final_exit_code = 0;
    REMUS_INIT_LOG();
    internal::MemoryPool::rdma_memory_resource memres = internal::MemoryPool::rdma_memory_resource(1 << 20);
    // -- Test 1, tiny objects -- //
    FakeObject* obj1 = memres.allocateT<FakeObject>();
    FakeObject* obj2 = memres.allocateT<FakeObject>();
    REMUS_TEST(aligned_to(obj1, 64), "Pointer should be aligned");
    REMUS_TEST(aligned_to(obj2, 64), "Pointer should be aligned");
    REMUS_TEST((uint64_t) obj1 > (uint64_t) obj2, "Addresses should be decreasing");
    REMUS_TEST(((uint64_t) obj1 - (uint64_t) obj2) >= 64, "Minimum alignment should be 64");
    if (exit_code == 0) REMUS_INFO("Passed Test 1!");
    else {
        REMUS_INFO("Failed Test 1");
        final_exit_code = exit_code;
        exit_code = 0;
    }

    // -- Test 2, allocating larger objects are spaced out enough -- //
    void* last_allocation = (void*) obj2;
    for(int i = 0; i < 20; i++){
        FakeObject64* obj64 = memres.allocateT<FakeObject64>();
        FakeObject128* obj128 = memres.allocateT<FakeObject128>();
        FakeObject256* obj256 = memres.allocateT<FakeObject256>();
        REMUS_TEST(aligned_to(obj64, 64), "Pointer should be aligned");
        REMUS_TEST(aligned_to(obj128, 64), "Pointer should be aligned");
        REMUS_TEST(aligned_to(obj256, 64), "Pointer should be aligned");
        REMUS_TEST(ptr_cmp(last_allocation, obj64) >= 64, "First compare, it={}", i);
        REMUS_TEST(ptr_cmp(obj64, obj128) >= 128, "Second compare, it={}", i);
        REMUS_TEST(ptr_cmp(obj128, obj256) >= 256, "Third compare, it={}", i);
        last_allocation = (void*) obj256;
    }
    if (exit_code == 0) REMUS_INFO("Passed Test 2!");
    else {
        REMUS_INFO("Failed Test 2");
        final_exit_code = exit_code;
        exit_code = 0;
    }

    // -- Test 3, error from slack message, in different ways -- //
    FakeObject64* x = memres.allocateT<FakeObject64>(3); // 192
    REMUS_TEST(aligned_to(x, 64), "Pointer should be aligned");
    memres.deallocateT(x, 3);
    FakeObject64* y = memres.allocateT<FakeObject64>(4); // 256
    REMUS_TEST(aligned_to(y, 64), "Pointer should be aligned");
    REMUS_INFO("last={} x={} y={}", (uint64_t) last_allocation, (uint64_t) x, (uint64_t) y);
    REMUS_TEST(ptr_cmp(last_allocation, y) >= 256, "Reallocating should still maintain distance to last pointer [1]"); // -- failing test

    last_allocation = y;
    FakeObject64* i = memres.allocateT<FakeObject64>(3); // 192
    REMUS_TEST(aligned_to(i, 64), "Pointer should be aligned");
    memres.deallocateT(x, 3);
    FakeObject256* j = memres.allocateT<FakeObject256>();
    REMUS_TEST(aligned_to(j, 64), "Pointer should be aligned");
    REMUS_TEST(ptr_cmp(last_allocation, j) >= 256, "Reallocating should still maintain distance to last pointer [2]"); // -- failing test
    if (exit_code == 0) REMUS_INFO("Passed Test 3!");
    else {
        REMUS_INFO("Failed Test 3");
        final_exit_code = exit_code;
        exit_code = 0;
    }
    return final_exit_code;
}
