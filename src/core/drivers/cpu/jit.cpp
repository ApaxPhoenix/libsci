#include "core/drivers/cpu/jit.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/bytecode/dtype.hpp"
#include "core/bytecode/code.hpp"
#include "core/drivers/cpu/jit/avx2.hpp"
#include "core/drivers/cpu/jit/neon.hpp"
#include "core/drivers/cpu/kernels/avx2.hpp"
#include "core/drivers/cpu/kernels/neon.hpp"
#include "core/drivers/cpu/kernels/table.hpp"
#include "core/types.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <print>
#include <utility>

namespace sci::core::drivers::cpu {

    // Below this many elements a call is over before compiled code would make a difference.
    constexpr size smallest = 256;

    // Weighted elements (count x chunk cost) at which a chunk compiles the first time it is seen:
    // compiling takes tens of microseconds, about what the kernels need for this much work.
    constexpr size worth = size{1} << 16;

    // Distinct chunks compiled at most; later ones keep running on the kernels.
    constexpr size limit = 4096;

    Jit::Jit() {
        if (const char* setting = std::getenv("SCI_JIT")) {
            enabled = std::strcmp(setting, "0") != 0;
            tracing = std::strcmp(setting, "trace") == 0;
        }
    }

    Jit& Jit::global() {
        static Jit jit;
        return jit;
    }

    size Jit::width(const kernels::Table& table, const Dtype type) noexcept {
        const size bytes = type == Dtype::float32 ? sizeof(float) : sizeof(double);
        if (&table == kernels::neon()) return 16 / bytes;
        if (&table == kernels::avx2()) return 32 / bytes;
        return 0;
    }

    // exp, log, sin and cos run best as kernel loops over whole blocks (a call per register costs
    // more than the temporaries it saves), so chunks that use them stay on the kernels.
    static bool compilable(const Chunk& chunk) noexcept {
        for (uint8 index = 0; index < chunk.length; ++index) {
            switch (chunk.code[index].opcode) {
                case Code::exp:
                case Code::log:
                case Code::sin:
                case Code::cos: return false;
                default: break;
            }
        }
        return true;
    }

    Jit::Function Jit::compile(const Chunk& chunk, const kernels::Table& table, const size count) {
        // A single instruction already runs as one kernel loop.
        if (!enabled || count < smallest || chunk.length < 2 || !chunk.valid() || width(table, chunk.type) == 0 || !compilable(chunk)) {
            return nullptr;
        }

        std::string key = chunk.signature();
        key += table.name;
        std::scoped_lock lock(guard);
        if (const auto found = functions.find(key); found != functions.end()) return found->second;
        if (functions.size() >= limit) return nullptr;
        if (++sightings[key] < 2 && count * chunk.cost() < worth) return nullptr;

        const Function function = build(chunk, table);
        sightings.erase(key);
        functions.emplace(std::move(key), function);
        if (tracing) {
            std::println(stderr, "[sci] jit: {} {} instructions for {} ({})", function ? "compiled" : "could not compile",
                         chunk.length, table.name, chunk.type == Dtype::float32 ? "float32" : "float64");
        }
        return function;
    }

    Jit::Function Jit::build(const Chunk& chunk, const kernels::Table& table) {
        if (!runtime) runtime.emplace();
        asmjit::CodeHolder code;
        if (code.init(runtime->environment(), runtime->cpu_features()) != asmjit::Error::kOk) return nullptr;
        const bool emitted = &table == kernels::neon() ? jit::neon(code, chunk) : jit::avx2(code, chunk);
        if (!emitted) return nullptr;
        Function function = nullptr;
        if (runtime->add(&function, &code) != asmjit::Error::kOk) return nullptr;
        return function;
    }

}
