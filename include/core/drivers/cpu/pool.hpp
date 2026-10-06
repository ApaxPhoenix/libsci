#pragma once

#include "core/types.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace sci::core::drivers::cpu {

    // Fork-join thread pool: run() hands out indices [0, count) to the workers and the calling
    // thread, and returns once every index has been processed.
    //
    // The hot path takes no locks. A job is published seqlock style: the generation is odd while the
    // job fields are rewritten, and workers announce themselves in `busy` before reading them. Idle
    // workers spin for a while before parking, so back-to-back jobs cost a few microseconds.
    class Pool {
    public:
        explicit Pool(size count = std::thread::hardware_concurrency());
        ~Pool();

        Pool(const Pool&) = delete;
        Pool& operator=(const Pool&) = delete;

        // Threads that can take part in a job, the caller included.
        [[nodiscard]] size width() const noexcept { return threads.size() + 1; }

        // At most `limit` threads (the caller included) take part; 0 means all of them.
        template <typename action>
        void run(const size count, action&& work, const size limit = 0) {
            using functor = std::remove_reference_t<action>;
            run(count, [](void* state, const size index) { (*static_cast<functor*>(state))(index); }, static_cast<void*>(&work), limit);
        }

        void run(size count, void (*routine)(void*, size), void* state, size limit);

        // Shared pool, sized from SCI_THREADS when set, otherwise from the hardware.
        [[nodiscard]] static Pool& global();

    private:
        void loop(size seat);
        void drain(void (*routine)(void*, size), void* state, size count) noexcept;

        std::vector<std::thread> threads;
        std::mutex exclusive;
        std::mutex guard;
        std::condition_variable signal;

        // The published job; only read by workers counted in `busy`.
        void (*task)(void*, size){nullptr};
        void* argument{nullptr};
        size total{0};
        size seats{0};

        alignas(128) std::atomic<uint64> generation{0};
        alignas(128) std::atomic<size> next{0};
        alignas(128) std::atomic<size> pending{0};
        alignas(128) std::atomic<size> busy{0};
        std::atomic<size> sleepers{0};
        std::atomic<bool> stopping{false};
    };

}
