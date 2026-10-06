#pragma once

#include "array/uninitialized.hpp"
#include "core/arch/alignment.hpp"
#include "core/drivers/device.hpp"
#include "core/drivers/memory.hpp"
#include "core/drivers/target.hpp"
#include "core/memory/arena.hpp"
#include "core/types.hpp"

#include <cstring>
#include <memory>
#include <new>

namespace sci::array {

    using core::drivers::Device;
    using core::drivers::Target;

    // The memory behind an NDArray: owned host memory, memory on a Device, or a slice of an Arena.
    template <typename element>
    class Storage {
    public:
        Storage() = default;

        // Zero-filled host memory.
        explicit Storage(const core::size total)
            : count(total), pointer(acquire(total)), owned(true) {
            std::uninitialized_value_construct_n(pointer, count);
        }

        Storage(const core::size total, Uninitialized)
            : count(total), pointer(acquire(total)), owned(true) {
            std::uninitialized_default_construct_n(pointer, count);
        }

        // Zero-filled memory on a device (GPU memory is host-mapped, so this is a plain memset).
        Storage(Device& device, const core::size total)
            : Storage(device, total, Uninitialized{}) {
            if (count > 0) std::memset(memory.host, 0, count * sizeof(element));
        }

        Storage(Device& device, const core::size total, Uninitialized)
            : count(total), owner(&device) {
            if (count > 0) owner->allocate(count * sizeof(element), memory);
        }

        // Zero-filled memory from an arena, which keeps ownership.
        Storage(core::memory::Arena& arena, const core::size total)
            : count(total), pointer(arena.allocate<element>(total)) {
            std::uninitialized_value_construct_n(pointer, count);
        }

        ~Storage() {
            release();
        }

        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;

        Storage(Storage&& storage) noexcept
            : memory(storage.memory), count(storage.count), owner(storage.owner), pointer(storage.pointer), owned(storage.owned) {
            storage.forget();
        }

        Storage& operator=(Storage&& storage) noexcept {
            if (this != &storage) {
                release();
                memory = storage.memory;
                count = storage.count;
                owner = storage.owner;
                pointer = storage.pointer;
                owned = storage.owned;
                storage.forget();
            }
            return *this;
        }

        [[nodiscard]] element* data() noexcept {
            return pointer ? pointer : static_cast<element*>(memory.host);
        }

        [[nodiscard]] const element* data() const noexcept {
            return pointer ? pointer : static_cast<const element*>(memory.host);
        }

        [[nodiscard]] core::size size() const noexcept { return count; }
        [[nodiscard]] Device* device() const noexcept { return owner; }
        [[nodiscard]] Target target() const noexcept { return owner ? memory.target : Target::cpu; }

    private:
        [[nodiscard]] static element* acquire(const core::size total) {
            if (total == 0) return nullptr;
            return static_cast<element*>(::operator new(total * sizeof(element), std::align_val_t{core::arch::line}));
        }

        void release() noexcept {
            if (owner && memory.bytes > 0) {
                owner->dispose(memory);
            } else if (owned && pointer) {
                std::destroy_n(pointer, count);
                ::operator delete(pointer, std::align_val_t{core::arch::line});
            }
            forget();
        }

        void forget() noexcept {
            memory = {};
            count = 0;
            owner = nullptr;
            pointer = nullptr;
            owned = false;
        }

        core::drivers::Memory memory{};
        core::size count{0};
        Device* owner{nullptr};
        element* pointer{nullptr};
        bool owned{false};
    };

}
