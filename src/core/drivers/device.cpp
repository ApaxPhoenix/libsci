#include "core/drivers/device.hpp"
#include "core/arch/alignment.hpp"
#include "core/assert.hpp"

#include <cstdlib>
#include <print>

namespace sci::core::drivers {

    void cpu::allocate(const size bytes, void** address) noexcept {
        if (!address) return;
        const size aligned = bytes + arch::cpu - 1 & ~(arch::cpu - 1);
        #if defined(_MSC_VER)
            *address = _aligned_malloc(aligned, arch::cpu);
        #else
            *address = std::aligned_alloc(arch::cpu, aligned);
        #endif
        ASSERT(*address != nullptr);
    }

    void cpu::dispose(void** address) noexcept {
        if (!address || !*address) return;
        #if defined(_MSC_VER)
            _aligned_free(*address);
        #else
            std::free(*address);
        #endif
        *address = nullptr;
    }

    void cpu::compose(std::span<const uint32>, void** const*, uint32, uint32, uint32, uint32) noexcept {
        std::println(stderr, "[sci::core::cpu] Warning: compose() is not implemented for CPU target.");
    }

    void cpu::sync() noexcept {
        std::println(stderr, "[sci::core::cpu] Warning: sync() is not implemented because CPU task queues are not yet managed.");
    }

    gpu::gpu(const vulkan::context& ctx)
        : context(ctx), allocator(context), dispatch(context), pipeline(context) {}

    gpu gpu::compose(const uint32 index) {
        return gpu(vulkan::context::compose(index));
    }

    void gpu::allocate(const size bytes, vulkan::buffer& object, void** address) noexcept {
        object = allocator.compose(bytes);
        ASSERT(object.host != nullptr);
        if (address) {
            *address = object.host;
        }
    }

    void gpu::dispose(vulkan::buffer& object, void** address) noexcept {
        vulkan::allocator::dispose(object);
        if (address) {
            *address = nullptr;
        }
    }

    void gpu::compose(const std::span<const uint32> code, const std::span<const vulkan::buffer> buffers, const uint32 x, const uint32 y, const uint32 z) {
        if (pipeline.compose(code, static_cast<uint32>(buffers.size()))) {
            dispatch.compose(pipeline, buffers, x, y, z);
        }
    }

    void gpu::sync() const noexcept {
        dispatch.wait();
    }

    bool gpu::ready() const noexcept {
        return dispatch.ready();
    }

    void gpu::dispose() noexcept {
        pipeline.dispose();
        context.dispose();
    }

    device::device(const drivers::target target, const uint32 index)
        : target_(target), index_(index) {
        if (target_ == target::gpu) {
            gpu_.emplace(gpu::compose(index_));
        }
    }

    device::~device() {
        if (target_ == target::gpu && gpu_.has_value()) {
            gpu_->dispose();
        }
    }

    void device::allocate(const size bytes, memory& object) {
        object.bytes = bytes;
        object.target = target_;
        object.handle = &object.buffer.host;

        if (target_ == target::gpu && gpu_.has_value()) {
            gpu_->allocate(bytes, object.buffer, object.handle);
        } else {
            cpu::allocate(bytes, object.handle);
        }
    }

    void device::dispose(memory& object) const noexcept {
        if (object.target == target::gpu && gpu_.has_value()) {
            gpu::dispose(object.buffer, object.handle);
        } else {
            cpu::dispose(object.handle);
        }
    }

    void device::compose(const std::span<const uint32> code, const std::span<const memory> items, const uint32 x, const uint32 y, const uint32 z) {
        const size count = items.size();
        ASSERT(count <= 16);

        if (target_ == target::gpu && gpu_.has_value()) {
            vulkan::buffer buffers[16]{};
            for (size i = 0; i < count; ++i) {
                buffers[i] = items[i].buffer;
            }
            gpu_->compose(code, std::span<const vulkan::buffer>(buffers, count), x, y, z);
        } else {
            void** pointers[16]{};
            for (size i = 0; i < count; ++i) {
                pointers[i] = items[i].handle;
            }
            cpu::compose(code, pointers, static_cast<uint32>(count), x, y, z);
        }
    }

    void device::sync() const noexcept {
        if (target_ == target::gpu && gpu_.has_value()) {
            gpu_->sync();
        } else {
            cpu::sync();
        }
    }

    bool device::ready() const noexcept {
        if (target_ == target::gpu && gpu_.has_value()) {
            return gpu_->ready();
        }
        return true;
    }

}