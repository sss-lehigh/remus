#include "mem_node.h"
namespace remus {
class ExtMemoryNode : public MemoryNode {
public:
  uint64_t total_cn_threads_;
  ExtMemoryNode(const MachineInfo &self, std::shared_ptr<ArgMap> args) : MemoryNode(self, args), total_cn_threads_(args->uget(remus::CN_THREADS) * (args->uget(remus::LAST_CN_ID) -
                           args->uget(remus::FIRST_CN_ID) + 1)) {}

  ~ExtMemoryNode() {
    // NB:  Spinning to await a remote write is safe here because (1) it's
    //      atomic, and (2) it's a write-once location
    auto cb_ptr =
        reinterpret_cast<internal::ControlBlock *>(this->segs_[0].seg_->raw());
    while (cb_ptr->control_flag_.load() < total_cn_threads_) {
      std::this_thread::yield();
    }
    REMUS_INFO("MemoryNode destructed");
  }
};
} // namespace remus
