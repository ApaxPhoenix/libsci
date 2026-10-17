#pragma once

#include "core/bytecode/chunk.hpp"
#include "core/drivers/vulkan/kernel.hpp"
#include "core/types.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace slang {
    struct IGlobalSession;
}

namespace sci::core::drivers::vulkan {

    struct Context;

    // Compiles chunks into shaders of their own: the shared code of shaders/common/ plus a `run`
    // generated for the chunk, in which every value is a named register, so there is no bytecode to
    // decode and no register array to index. Compiling (Slang, then the driver) takes milliseconds,
    // so it happens on a background thread: a chunk runs on the interpreter until its kernel is
    // ready, and on the kernel from then on. Float32 only, like the interpreter.
    //
    // SCI_JIT=0 turns the compiler off and SCI_JIT=trace reports every compile.
    class Compiler {
    public:
        // `bytes`: size of the push-constant block the shaders take.
        Compiler(const Context& context, uint32 bytes);
        ~Compiler();

        Compiler(const Compiler&) = delete;
        Compiler& operator=(const Compiler&) = delete;

        // The chunk's kernel once it is compiled. Null before that (the first call queues the
        // compile) and when it cannot be compiled.
        [[nodiscard]] const Kernel* find(const Chunk& chunk);

    private:
        // A shader for one chunk: shaders/common/ around a `run` in which every value is a named float4.
        [[nodiscard]] static std::string generate(const Chunk& chunk);

        // SPIR-V for each entry point of a generated shader, compiled with the options CMake gives
        // slangc. Empty on failure, with Slang's messages in `diagnostics`.
        [[nodiscard]] static std::vector<std::vector<uint32>> compile(slang::IGlobalSession* global, const std::string& source, std::string& diagnostics);

        // The compiler thread: takes jobs off the queue until the Compiler is destroyed.
        void loop();

        const Context& context;
        uint32 bytes{0};
        std::atomic<bool> enabled{true};
        bool tracing{false};

        std::mutex guard;
        std::condition_variable signal;

        // Kernels by chunk signature: null while queued or compiling, and when compiling failed.
        std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels;

        // Chunks waiting for the compiler thread: signature and generated shader source.
        std::deque<std::pair<std::string, std::string>> jobs;

        bool stopping{false};
        std::thread worker;
    };

}
