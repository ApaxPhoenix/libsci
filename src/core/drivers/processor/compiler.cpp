#include "core/drivers/processor/compiler.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/bytecode/precision.hpp"
#include "core/bytecode/code.hpp"
#include "core/drivers/processor/compiler/avx2.hpp"
#include "core/drivers/processor/compiler/neon.hpp"
#include "core/drivers/processor/kernels/avx2.hpp"
#include "core/drivers/processor/kernels/neon.hpp"
#include "core/drivers/processor/kernels/table.hpp"
#include "core/types.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <print>
#include <utility>

namespace sci::core::drivers::processor {

    // Below this many elements a call is over before compiled code would make a difference.
    constexpr size smallest = 256;

    // Weighted elements (count x chunk cost) at which a chunk compiles the first time it is seen:
    // compiling takes tens of microseconds, about what the kernels need for this much work.
    constexpr size worth = size{1} << 16;

    // Distinct chunks compiled at most; later ones keep running on the kernels.
    constexpr size limit = 4096;

    Compiler::Compiler() {
        if (const char* setting = std::getenv("SCI_JIT")) {
            enabled = std::strcmp(setting, "0") != 0;
            tracing = std::strcmp(setting, "trace") == 0;
        }
    }

    Compiler& Compiler::global() {
        static Compiler compiler;
        return compiler;
    }

    size Compiler::width(const kernels::Table& table, const Precision precision) noexcept {
        const size bytes = precision == Precision::single ? sizeof(float) : sizeof(double);
        if (&table == kernels::neon) return 16 / bytes;
        if (&table == kernels::avx2) return 32 / bytes;
        return 0;
    }

    Compiler::Function Compiler::compile(const Chunk& chunk, const kernels::Table& table, const size count) {
        // A single instruction already runs as one kernel loop.
        if (!enabled || count < smallest || chunk.length < 2 || !chunk.valid() || width(table, chunk.precision) == 0) {
            return nullptr;
        }

        // The exponential, logarithm, circular and hyperbolic functions (codes exponential through
        // hyperbolic::arctangent) run best as kernel loops over whole blocks: a call per register
        // costs more than the temporaries it saves, so chunks that use them stay on the kernels.
        for (uint8 index = 0; index < chunk.length; ++index) {
            if (const Code code = chunk.code[index].code; code >= Code::exponential && code <= hyperbolic::arctangent) return nullptr;
        }

        std::string key = chunk.signature();
        key += table.name;
        std::scoped_lock lock(guard);
        if (const auto found = functions.find(key); found != functions.end()) return found->second;
        if (functions.size() >= limit) return nullptr;
        if (++sightings[key] < 2 && count * chunk.cost() < worth) return nullptr;

        if (!runtime) runtime.emplace();
        Function function = nullptr;
        asmjit::CodeHolder code;
        if (code.init(runtime->environment(), runtime->cpu_features()) == asmjit::Error::kOk &&
            (&table == kernels::neon ? compiler::neon(code, chunk) : compiler::avx2(code, chunk)) &&
            runtime->add(&function, &code) != asmjit::Error::kOk) {
            function = nullptr;
        }
        sightings.erase(key);
        functions.emplace(std::move(key), function);
        if (tracing) {
            std::println(stderr, "[sci] jit: {} {} instructions for {} ({})", function ? "compiled" : "could not compile",
                         chunk.length, table.name, chunk.precision == Precision::single ? "float32" : "float64");
        }
        return function;
    }

}
