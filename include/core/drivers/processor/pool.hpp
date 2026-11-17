#pragma once

/// @file
/// @brief Fork-join thread pool.

#include "core/types.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace sci::core::drivers::processor {

    /// Fork-join thread pool: run() hands out indices [0, count) to the workers and the calling
    /// thread, and returns once every index has been processed.
    ///
    /// The hot path takes no locks. A job is published seqlock style: the generation is odd while the
    /// job fields are rewritten, and workers announce themselves in `busy` before reading them. Idle
    /// workers spin for a while before parking, so back-to-back jobs cost a few microseconds.
    class Pool {
    public:
        /// @brief A pool of threads, the caller among them.
        /// @param count How many threads take part, the caller included.
        explicit Pool(size count = std::thread::hardware_concurrency());
        ~Pool();

        Pool(const Pool &) = delete;
        Pool &operator=(const Pool &) = delete;

        /// Threads that can take part in a job, the caller included.
        /// @return The width.
        [[nodiscard]] size width() const noexcept { return threads.size() + 1; }

        /// At most `limit` threads (the caller included) take part; 0 means all of them.
        /// @param count How many indices.
        /// @param work Called with each index.
        /// @param limit The most threads that take part.
        template <typename action> void run(const size count, action &&work, const size limit = 0) {
            using callable = std::remove_reference_t<action>;
            run(
                    count, [](void *state, const size index) { (*static_cast<callable *>(state))(index); },
                    static_cast<void *>(&work), limit);
        }

        /// @brief Runs a routine over every index, across the pool.
        /// @param count How many indices.
        /// @param routine Called with the state and each index.
        /// @param state Handed to the routine.
        /// @param limit The most threads that take part; 0 for all of them.
        void run(size count, void (*routine)(void *, size), void *state, size limit);

        /// Shared pool, sized from SCI_THREADS when set, otherwise from the hardware.
        /// @return The pool.
        [[nodiscard]] static Pool &global();

    private:
        /// Pauses the core for a moment inside a spin loop.
        static void relax() noexcept;

        /// Spins on a condition, then starts yielding the core so a preempted peer can finish.
        /// @param ready The condition.
        template <typename predicate> static void await(predicate &&ready) noexcept;

        /// @brief A worker's life: waits for jobs and takes part in them.
        /// @param seat The worker's seat.
        void loop(size seat);
        /// @brief Takes indices off the job until none are left.
        /// @param routine The job's routine.
        /// @param state Its state.
        /// @param count How many indices.
        void drain(void (*routine)(void *, size), void *state, size count) noexcept;

        /// @brief The workers.
        std::vector<std::thread> threads;
        /// @brief Lets one job run at a time.
        std::mutex exclusive;
        /// @brief Guards parking and waking.
        std::mutex guard;
        /// @brief Wakes parked workers.
        std::condition_variable signal;

        /// The published job; only read by workers counted in `busy`.
        void (*task)(void *, size){nullptr};
        /// @brief The job's state.
        void *argument{nullptr};
        /// @brief How many indices the job has.
        size total{0};
        /// @brief How many threads may take part in the job.
        size seats{0};

        /// @brief Odd while a job is published, and bumped for each one.
        alignas(128) std::atomic<uint64> generation{0};
        /// @brief The next index to hand out.
        alignas(128) std::atomic<size> next{0};
        /// @brief How many indices are not finished.
        alignas(128) std::atomic<size> pending{0};
        /// @brief How many workers are taking part.
        alignas(128) std::atomic<size> busy{0};
        /// @brief How many workers are parked.
        std::atomic<size> sleepers{0};
        /// @brief Set when the pool is being destroyed.
        std::atomic<bool> stopping{false};
    };

}
