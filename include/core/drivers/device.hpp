#pragma once

#include "core/drivers/vulkan/allocator.hpp"
#include "core/drivers/vulkan/context.hpp"
#include "core/drivers/vulkan/dispatch.hpp"
#include "core/drivers/vulkan/pipeline.hpp"
#include "core/types.hpp"

#include <optional>
#include <span>

namespace sci::core::drivers {

    enum class target : uint32 {
        cpu,
        gpu
    };

    struct memory {
        void** handle{nullptr};
        vulkan::buffer buffer{};
        size bytes{0};
        target target{target::cpu};
    };

    struct cpu {
        static void allocate(size bytes, void** address) noexcept;
        static void dispose(void** address) noexcept;
        static void compose(std::span<const uint32> code, void** const* items, uint32 count, uint32 x, uint32 y = 1, uint32 z = 1) noexcept;
        static void sync() noexcept;
    };

    struct gpu {
        vulkan::context context;
        vulkan::allocator allocator;
        vulkan::dispatch dispatch;
        vulkan::pipeline pipeline;

        explicit gpu(const vulkan::context& ctx);

        static gpu compose(uint32 index = 0);

        void allocate(size bytes, vulkan::buffer& object, void** address) noexcept;
        static void dispose(vulkan::buffer& object, void** address) noexcept;
        void compose(std::span<const uint32> code, std::span<const vulkan::buffer> buffers, uint32 x, uint32 y = 1, uint32 z = 1);
        void sync() const noexcept;
        [[nodiscard]] bool ready() const noexcept;
        void dispose() noexcept;
    };

    class device {
    public:
        explicit device(target target = target::gpu, uint32 index = 0);
        ~device();

        [[nodiscard]] target target() const noexcept { return target_; }
        [[nodiscard]] uint32 index() const noexcept { return index_; }

        void allocate(size bytes, memory& object);
        void dispose(memory& object) const noexcept;
        void compose(std::span<const uint32> code, std::span<const memory> items, uint32 x, uint32 y = 1, uint32 z = 1);
        void sync() const noexcept;
        [[nodiscard]] bool ready() const noexcept;

    private:
        drivers::target target_{target::cpu};
        uint32 index_{0};
        std::optional<gpu> gpu_{std::nullopt};
    };

}