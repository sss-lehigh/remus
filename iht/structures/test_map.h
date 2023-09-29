#pragma once

#include <infiniband/verbs.h>
#include <cstdint>
#include <atomic>

#include "../../rdma/channel/sync_accessor.h"
#include "../../rdma/connection_manager/connection.h"
#include "../../rdma/connection_manager/connection_manager.h"
#include "../../rdma/memory_pool/memory_pool.h"
#include "../../rdma/rdma_memory.h"
#include "../../logging/logging.h"
#include "../common.h"

#define LEN 8

using ::rome::rdma::ConnectionManager;
using ::rome::rdma::MemoryPool;
using ::rome::rdma::remote_nullptr;
using ::rome::rdma::remote_ptr;
using ::rome::rdma::RemoteObjectProto;

template<class K, class V>
class TestMap {
private:
    MemoryPool::Peer self_;

    struct alignas(64) List {
        uint64_t vals[LEN];
    };

    inline void InitList(remote_ptr<List> p){
        for (size_t i = 0; i < LEN; i++){
            p->vals[i] = i;
        }
    }

    remote_ptr<List> root;  // Start of list

    template <typename T>
    inline bool is_local(remote_ptr<T> ptr){
        return ptr.id() == self_.id;
    }

    template <typename T>
    inline bool is_null(remote_ptr<T> ptr){
        return ptr == remote_nullptr;
    }

public:
    MemoryPool* pool_;

    using conn_type = MemoryPool::conn_type;

    TestMap(MemoryPool::Peer self, MemoryPool* pool) : self_(self), pool_(pool){};

    /// @brief Initialize the IHT by connecting to the peers and exchanging the PList pointer
    /// @param host the leader of the initialization
    /// @param peers all the nodes in the neighborhood
    /// @return status code for the function
    absl::Status Init(MemoryPool::Peer host, const std::vector<MemoryPool::Peer> &peers) {
        bool is_host_ = self_.id == host.id;

        if (is_host_){
            // Host machine, it is my responsibility to initiate configuration
            RemoteObjectProto proto;
            remote_ptr<List> iht_root = pool_->Allocate<List>();
            // Init plist and set remote proto to communicate its value
            InitList(iht_root);
            this->root = iht_root;
            proto.set_raddr(iht_root.address());

            // Iterate through peers
            for (auto p = peers.begin(); p != peers.end(); p++){
                // Ignore sending pointer to myself
                if (p->id == self_.id) continue;

                // Form a connection with the machine
                auto conn_or = pool_->connection_manager()->GetConnection(p->id);
                ROME_CHECK_OK(ROME_RETURN(conn_or.status()), conn_or);

                // Send the proto over
                absl::Status status = conn_or.value()->channel()->Send(proto);
                ROME_CHECK_OK(ROME_RETURN(status), status);
            }
        } else {
            // Listen for a connection
            auto conn_or = pool_->connection_manager()->GetConnection(host.id);
            ROME_CHECK_OK(ROME_RETURN(conn_or.status()), conn_or);

            // Try to get the data from the machine, repeatedly trying until successful
            auto got = conn_or.value()->channel()->TryDeliver<RemoteObjectProto>();
            while(got.status().code() == absl::StatusCode::kUnavailable) {
                got = conn_or.value()->channel()->TryDeliver<RemoteObjectProto>();
            }
            ROME_CHECK_OK(ROME_RETURN(got.status()), got);

            // From there, decode the data into a value
            remote_ptr<List> iht_root = decltype(iht_root)(host.id, got->raddr());
            this->root = iht_root;
        }

        return absl::OkStatus();
    }


    /// @brief Gets a value at the key.
    /// @param key the key to search on
    /// @return if the key was found or not. The value at the key is stored in RdmaIHT::result
    HT_Res<V> contains(K key){
        remote_ptr<List> prealloc = pool_->Allocate<List>();
        List temp = *std::to_address(prealloc);
        temp.vals[0] = 100;
        *std::to_address(prealloc) = temp;
        remote_ptr<List> list = pool_->Read<List>(this->root, prealloc);
        if (list.address() != prealloc.address()){
            ROME_INFO("Prealloc not working as expected");
        }
        List l = *std::to_address(list);
        for(int i = 0; i < LEN; i++){
            if(l.vals[i] != i){
                ROME_INFO("Illegal inequality {} {}", l.vals[i], i);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        pool_->Deallocate<List>(list);

        /*
        List old = *std::to_address(this->root);
        uint64_t test = old.vals[0];
        old.vals[0]++;
        pool_->Write<List>(this->root, old);
        List l = *std::to_address(this->root);
        if(l.vals[0] == test){
            ROME_INFO("Illegal equality");
        }
        */

        return HT_Res<V>(TRUE_STATE, key);
    }
    
    /// @brief Insert a key and value into the iht. Result will become the value at the key if already present.
    /// @param key the key to insert
    /// @param value the value to associate with the key
    /// @return if the insert was successful
    HT_Res<V> insert(K key, V value){
        return HT_Res<V>(TRUE_STATE, 0);
    }
    
    /// @brief Will remove a value at the key. Will stored the previous value in result.
    /// @param key the key to remove at
    /// @return if the remove was successful
    HT_Res<V> remove(K key){
        return HT_Res<V>(FALSE_STATE, 0);
    }

    /// Function signature added to match map interface. No intermediary cleanup necessary so unusued
    void try_rehash(){
        // Unused function b/c no cleanup necessary
    }

    /// @brief Populate only works when we have numerical keys. Will add data
    /// @param count the number of values to insert. Recommended in total to do key_range / 2
    /// @param key_lb the lower bound for the key range
    /// @param key_ub the upper bound for the key range
    /// @param value the value to associate with each key. Currently, we have asserts for result to be equal to the key. Best to set value equal to key!
    void populate(int op_count, K key_lb, K key_ub, std::function<K(V)> value){
        // Populate only works when we have numerical keys
        K key_range = key_ub - key_lb;

        // Create a random operation generator that is 
        // - evenly distributed among the key range  
        std::uniform_real_distribution<double> dist = std::uniform_real_distribution<double>(0.0, 1.0);
        std::default_random_engine gen((unsigned) std::time(NULL));
        for (int c = 0; c < op_count; c++){
            int k = dist(gen) * key_range + key_lb;
            insert(k, value(k));
            // Wait some time before doing next insert...
            std::this_thread::sleep_for(std::chrono::nanoseconds(10));
        }
    }
};
