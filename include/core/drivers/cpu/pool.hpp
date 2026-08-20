#pragma once

#include "core/types.hpp"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace sci::core::drivers::cpu {

    class pool {
    public:
        explicit pool(size count = std::thread::hardware_concurrency());
        ~pool();

        template <typename action>
        void submit(action&& work) {
            {
                std::unique_lock lock(guard);
                tasks.emplace(std::forward<action>(work));
            }
            signal.notify_one();
        }

    private:
        std::vector<std::thread> threads{};
        std::queue<std::function<void()>> tasks{};
        std::mutex guard{};
        std::condition_variable signal{};
        bool stop{false};
    };

}