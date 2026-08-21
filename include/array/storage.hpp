#pragma once

#include "core/drivers/device.hpp"
#include "core/memory/arena.hpp"
#include "core/types.hpp"

namespace sci::array {

template <typename element>
class storage {
public:
    storage() = default;

    explicit storage(const core::size total)
        : count_(total), pointer_(new element[total]()), owned_(true) {}

    explicit storage(core::drivers::device& device, const core::size total)
        : count_(total), device_(&device), target_(device.target()) {
        if (count_ == 0) return;
        device_->allocate(count_ * sizeof(element), memory_);
    }

    explicit storage(core::memory::arena& arena, const core::size total)
        : count_(total), pointer_(arena.allocate<element>(total)) {}

    ~storage() {
        release();
    }

    storage(const storage&) = delete;
    storage& operator=(const storage&) = delete;

    storage(storage&& other) noexcept
        : memory_(other.memory_),
          count_(other.count_),
          device_(other.device_),
          pointer_(other.pointer_),
          owned_(other.owned_),
          target_(other.target_) {
        other.memory_ = {};
        other.count_ = 0;
        other.device_ = nullptr;
        other.pointer_ = nullptr;
        other.owned_ = false;
        other.target_ = core::drivers::target::cpu;
    }

    storage& operator=(storage&& other) noexcept {
        if (this != &other) {
            release();
            memory_ = other.memory_;
            count_ = other.count_;
            device_ = other.device_;
            pointer_ = other.pointer_;
            owned_ = other.owned_;
            target_ = other.target_;

            other.memory_ = {};
            other.count_ = 0;
            other.device_ = nullptr;
            other.pointer_ = nullptr;
            other.owned_ = false;
            other.target_ = core::drivers::target::cpu;
        }
        return *this;
    }

    [[nodiscard]] element* data() noexcept {
        if (pointer_) return pointer_;
        if (memory_.handle) return static_cast<element*>(*memory_.handle);
        return nullptr;
    }

    [[nodiscard]] const element* data() const noexcept {
        if (pointer_) return pointer_;
        if (memory_.handle) return static_cast<const element*>(*memory_.handle);
        return nullptr;
    }

    [[nodiscard]] core::size size() const noexcept { return count_; }
    [[nodiscard]] const core::drivers::memory& handle() const noexcept { return memory_; }
    [[nodiscard]] core::drivers::device* device() const noexcept { return device_; }
    [[nodiscard]] core::drivers::target target() const noexcept { return target_; }

private:
    void release() noexcept {
        if (device_ && memory_.bytes > 0) {
            device_->dispose(memory_);
        } else if (owned_ && pointer_) {
            delete[] pointer_;
        }
        pointer_ = nullptr;
        device_ = nullptr;
        memory_ = {};
        count_ = 0;
    }

    core::drivers::memory memory_{};
    core::size count_{0};
    core::drivers::device* device_{nullptr};
    element* pointer_{nullptr};
    bool owned_{false};
    core::drivers::target target_{core::drivers::target::cpu};
};

}