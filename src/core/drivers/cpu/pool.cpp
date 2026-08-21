#include "core/drivers/cpu/pool.hpp"

namespace sci::core::drivers::cpu {

    pool::pool(const size count) {
        threads.reserve(count);
        for (size index = 0; index < count; ++index) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(guard);
                        signal.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    pool::~pool() {
        {
            std::unique_lock lock(guard);
            stop = true;
        }
        signal.notify_all();
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
    }

}