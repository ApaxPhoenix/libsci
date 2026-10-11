#include "core/drivers/processor/pool.hpp"

#include "core/types.hpp"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(_M_ARM64)
#include <intrin.h>
#endif

namespace sci::core::drivers::processor {

    // Set on pool threads and on a caller while it helps: nested run() calls execute inline.
    static thread_local bool nested = false;

    // Polls before an idle worker parks on the condition variable (a few hundred microseconds).
    constexpr size patience = size{1} << 17;

    void Pool::relax() noexcept {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            _mm_pause();
        #elif defined(__aarch64__)
            __asm__ __volatile__("yield");
        #elif defined(_M_ARM64)
            __yield();
        #endif
    }

    template <typename predicate>
    void Pool::await(predicate&& ready) noexcept {
        for (size spin = 0; !ready(); ++spin) {
            if (spin < 4096) {
                relax();
            } else {
                std::this_thread::yield();
            }
        }
    }

    Pool::Pool(const size count) {
        const size workers = count > 1 ? count - 1 : 0;
        threads.reserve(workers);
        for (size seat = 1; seat <= workers; ++seat) {
            threads.emplace_back([this, seat] { loop(seat); });
        }
    }

    Pool::~Pool() {
        stopping.store(true);
        generation.fetch_add(2);
        {
            std::scoped_lock lock(guard);
        }
        signal.notify_all();
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
    }

    Pool& Pool::global() {
        static Pool instance([] {
            if (const char* text = std::getenv("SCI_THREADS")) {
                if (const long value = std::strtol(text, nullptr, 10); value > 0) {
                    return static_cast<size>(value);
                }
            }
            const size count = std::thread::hardware_concurrency();
            return count > 0 ? count : size{1};
        }());
        return instance;
    }

    void Pool::run(const size count, void (*routine)(void*, size), void* state, const size limit) {
        if (count == 0) return;

        if (count == 1 || threads.empty() || nested || limit == 1) {
            for (size index = 0; index < count; ++index) {
                routine(state, index);
            }
            return;
        }

        std::scoped_lock serial(exclusive);

        // Odd generation: the job fields are being rewritten. Wait out any worker still reading the
        // previous job before touching them.
        generation.fetch_add(1);
        await([this] { return busy.load() == 0; });

        task = routine;
        argument = state;
        total = count;
        seats = limit == 0 ? width() : limit;
        next.store(0, std::memory_order_relaxed);
        pending.store(count, std::memory_order_relaxed);
        generation.fetch_add(1);

        if (sleepers.load() > 0) {
            {
                std::scoped_lock lock(guard);
            }
            signal.notify_all();
        }

        nested = true;
        drain(routine, state, count);
        nested = false;

        await([this] { return pending.load(std::memory_order_acquire) == 0; });
    }

    void Pool::loop(const size seat) {
        nested = true;
        uint64 seen = 0;

        while (true) {
            uint64 current = generation.load(std::memory_order_acquire);
            for (size spin = 0; (current == seen || (current & 1) != 0) && spin < patience; ++spin) {
                relax();
                current = generation.load(std::memory_order_acquire);
            }

            if (current == seen || (current & 1) != 0) {
                std::unique_lock lock(guard);
                sleepers.fetch_add(1);
                signal.wait(lock, [&] {
                    current = generation.load();
                    return stopping.load() || (current != seen && (current & 1) == 0);
                });
                sleepers.fetch_sub(1);
            }

            if (stopping.load()) return;

            busy.fetch_add(1);
            if (generation.load() != current) {
                busy.fetch_sub(1);
                continue;
            }

            seen = current;
            if (seat < seats) {
                drain(task, argument, total);
            }
            busy.fetch_sub(1, std::memory_order_release);
        }
    }

    void Pool::drain(void (*routine)(void*, size), void* state, const size count) noexcept {
        for (size index = next.fetch_add(1, std::memory_order_relaxed); index < count; index = next.fetch_add(1, std::memory_order_relaxed)) {
            routine(state, index);
            pending.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

}
