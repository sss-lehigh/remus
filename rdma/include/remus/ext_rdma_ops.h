#pragma once
namespace remus::internal {
// this is a special version that allows specify the size of the write
template <typename T>
void Write(rdma_ptr<T> ptr, const T &val, uint8_t *seg, int32_t rkey,
           int32_t lkey, Connection *conn, std::atomic<int> *ack, size_t size) {
  T *local = (T *)seg;

  REMUS_ASSERT((uint64_t)local != ptr.address(), "WTF");
  // [mfs] Is this memset really necessary?  Maybe for gaps in structs?
  std::memcpy(local, &val, size);
  ibv_sge sge; // only 3 fields, so no need to memset it to 0
  sge.addr = reinterpret_cast<uint64_t>(local);
  sge.length = size;
  sge.lkey = lkey;

  ibv_send_wr send_wr_{}; // [mfs] Should we memset this?  It's large.
  send_wr_.wr_id = (uint64_t)ack;
  send_wr_.num_sge = 1;
  send_wr_.sg_list = &sge;
  send_wr_.opcode = IBV_WR_RDMA_WRITE;
  send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
  send_wr_.wr.rdma.remote_addr = ptr.address();
  send_wr_.wr.rdma.rkey = rkey;

  // set the counter to the # of completions we expect, then make request
  *ack = 1;
  conn->send_onesided(&send_wr_);
  // Poll until we get an ack for this event, and assert it's good
  ibv_wc wc;
  while (*ack != 0) {
    int poll = conn->poll_cq(1, &wc);
    if (poll == 0 || (poll < 0 && errno == EAGAIN))
      continue;
    if (wc.status != IBV_WC_SUCCESS) {
      REMUS_ASSERT(wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} ({})",
                   (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                   (std::stringstream() << ptr).str());
    }
    int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
    REMUS_ASSERT(old >= 1, "Broken synchronization");
  }
}

// this is a special version that allows specify the size of the write,
// the seg is already set up
template <typename T>
void Write(rdma_ptr<T> ptr, uint8_t *seg, int32_t rkey, int32_t lkey,
           Connection *conn, std::atomic<int> *ack, size_t size) {
  T *local = (T *)seg;
  REMUS_ASSERT((uint64_t)local != ptr.address(), "WTF");
  ibv_sge sge; // only 3 fields, so no need to memset it to 0
  sge.addr = reinterpret_cast<uint64_t>(local);
  sge.length = size;
  sge.lkey = lkey;

  ibv_send_wr send_wr_{}; // [mfs] Should we memset this?  It's large.
  send_wr_.wr_id = (uint64_t)ack;
  send_wr_.num_sge = 1;
  send_wr_.sg_list = &sge;
  send_wr_.opcode = IBV_WR_RDMA_WRITE;
  send_wr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
  send_wr_.wr.rdma.remote_addr = ptr.address();
  send_wr_.wr.rdma.rkey = rkey;

  // set the counter to the # of completions we expect, then make request
  *ack = 1;
  conn->send_onesided(&send_wr_);
  // Poll until we get an ack for this event, and assert it's good
  ibv_wc wc;
  while (*ack != 0) {
    int poll = conn->poll_cq(1, &wc);
    if (poll == 0 || (poll < 0 && errno == EAGAIN))
      continue;
    if (wc.status != IBV_WC_SUCCESS) {
      REMUS_ASSERT(wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} ({})",
                   (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
                   (std::stringstream() << ptr).str());
    }
    int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
    REMUS_ASSERT(old >= 1, "Broken synchronization");
  }
}

template <typename T>
std::vector<T> BatchRead(std::vector<rdma_ptr<T>> &addresses, uint8_t *seg,
                         int32_t rkey, int32_t lkey, Connection *conn,
                         std::atomic<int> *ack) {
  // Calculate the number of work requests and the number of requests per
  // batch
  const size_t num_requests = addresses.size();
  const size_t num_wrs_per_batch = kMaxWr;

  T *local = (T *)seg;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  // Create work requests and SGEs for each remote address
  ibv_send_wr wrs[num_wrs_per_batch];
  ibv_sge sges[num_wrs_per_batch];
#pragma clang diagnostic pop
  // Construct return vector from local buffer
  std::vector<T> result;
  result.reserve(num_requests);
  // Set up work requests - one WR per remote address in batches
  for (size_t i = 0;
       i < (num_requests + num_wrs_per_batch - 1) / num_wrs_per_batch; i++) {
    size_t j = 0;
    for (; j < num_wrs_per_batch && i * num_wrs_per_batch + j < num_requests;
         j++) {
      sges[j].addr = reinterpret_cast<uint64_t>(local) + j * sizeof(T);
      sges[j].length = sizeof(T);
      sges[j].lkey = lkey;

      // Set up work request
      wrs[j].wr_id = (uint64_t)ack;
      wrs[j].opcode = IBV_WR_RDMA_READ;
      wrs[j].send_flags = IBV_SEND_FENCE;
      if (j == num_wrs_per_batch - 1 ||
          i * num_wrs_per_batch + j == num_requests - 1)
        wrs[j].send_flags |= IBV_SEND_SIGNALED;
      wrs[j].num_sge = 1;
      wrs[j].sg_list = &sges[j];
      wrs[j].next = (j < num_wrs_per_batch - 1 &&
                             i * num_wrs_per_batch + j < num_requests - 1
                         ? &wrs[j + 1]
                         : nullptr);
      wrs[j].wr.rdma.remote_addr =
          addresses[i * num_wrs_per_batch + j].address();
      wrs[j].wr.rdma.rkey = rkey;
    }

    // Set the counter to the expected completion (one per batch)
    *ack = 1;
    conn->send_onesided(wrs);

    // Poll until the work completion is received
    ibv_wc wc;
    while (*ack != 0) {
      int poll = conn->poll_cq(1, &wc);
      if (poll == 0 || (poll < 0 && errno == EAGAIN))
        continue;
      // [mfs] Why doesn't write look for poll != 1, but this does?
      if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
        REMUS_ASSERT(
            poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} @ {}",
            (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
            format_rdma_ptr(
                addresses[0])); // [ys] this addresses[0] doesn't really
                                // make sense, just for debugging
      }
      int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
      REMUS_ASSERT(old >= 1, "Broken synchronization");
    }

    for (size_t k = 0; k < j; k++) {
      result.push_back(local[k]);
    }
  }
  return result;
}

/// to simplify the logic and allow user do optimization, all addresses's remote
/// pointers should be on the same node
///
/// @tparam T
/// @param addresses
/// @param values
/// @param size
template <typename T>
void BatchWrite(std::vector<rdma_ptr<T>> &addresses, const std::vector<T> &values,
                bool use_one, uint8_t *seg, int32_t rkey, int32_t lkey,
                Connection *conn, std::atomic<int> *ack) {
  // Get connection info for the remote memory pool using the first remote
  // pointer
  //
  const size_t num_requests = addresses.size();
  const size_t num_wrs_per_batch = kMaxWr;
  T *local = (T *)seg;
  // Create work requests and SGEs for each write operation
  ibv_send_wr wrs[num_wrs_per_batch];
  ibv_sge sges[num_wrs_per_batch];

  // Set up work requests in batches similar to BatchRead
  for (size_t i = 0;
       i < (num_requests + num_wrs_per_batch - 1) / num_wrs_per_batch; i++) {
    if (!use_one) {
      if (i != (num_requests + num_wrs_per_batch - 1) / num_wrs_per_batch - 1) {
        memcpy(local, values.data() + i * num_wrs_per_batch,
               num_wrs_per_batch * sizeof(T));
      } else {
        memcpy(local, values.data() + i * num_wrs_per_batch,
               (num_requests - i * num_wrs_per_batch) * sizeof(T));
      }
    }
    size_t j = 0;
    for (; j < num_wrs_per_batch && i * num_wrs_per_batch + j < num_requests;
         j++) {
      auto remote_ptr = addresses[i * num_wrs_per_batch + j];
      if (use_one) {
        sges[j].addr = reinterpret_cast<uint64_t>(local);
      } else {
        sges[j].addr = reinterpret_cast<uint64_t>(local) + j * sizeof(T);
      }
      sges[j].length = sizeof(T);
      sges[j].lkey = lkey;

      wrs[j].wr_id = (uint64_t)ack;
      wrs[j].opcode = IBV_WR_RDMA_WRITE;
      wrs[j].send_flags = IBV_SEND_FENCE;
      if (j == num_wrs_per_batch - 1 ||
          i * num_wrs_per_batch + j == num_requests - 1)
        wrs[j].send_flags |= IBV_SEND_SIGNALED;
      wrs[j].num_sge = 1;
      wrs[j].sg_list = &sges[j];
      wrs[j].next = (j < num_wrs_per_batch - 1 &&
                             i * num_wrs_per_batch + j < num_requests - 1
                         ? &wrs[j + 1]
                         : nullptr);
      wrs[j].wr.rdma.remote_addr = remote_ptr.address();
      wrs[j].wr.rdma.rkey = rkey;
    }

    // Post the batch: set the counter to 1 since only the last WR is signaled
    *ack = 1;
    conn->send_onesided(wrs);

    // Poll until the work completion is received
    ibv_wc wc;
    while (*ack != 0) {
      int poll = conn->poll_cq(1, &wc);
      if (poll == 0 || (poll < 0 && errno == EAGAIN))
        continue;
      if (poll != 1 || wc.status != IBV_WC_SUCCESS) {
        REMUS_ASSERT(
            poll == 1 && wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} @ {}",
            (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
            format_rdma_ptr(addresses[0]));
      }
      int old = ((std::atomic<int> *)wc.wr_id)->fetch_sub(1);
      REMUS_ASSERT(old >= 1, "Broken synchronization");
    }
  }
}
} // namespace remus::internal