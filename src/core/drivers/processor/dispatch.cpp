#include "core/drivers/processor/dispatch.hpp"
#include "core/drivers/processor/context.hpp"
#include "core/drivers/processor/compiler.hpp"
#include "core/drivers/processor/kernels/avx2.hpp"
#include "core/drivers/processor/kernels/avx512.hpp"
#include "core/drivers/processor/kernels/fallback.hpp"
#include "core/drivers/processor/kernels/neon.hpp"
#include "core/drivers/processor/kernels/routines.hpp"
#include "core/drivers/processor/kernels/table.hpp"
#include "core/drivers/processor/plan.hpp"
#include "core/drivers/processor/pool.hpp"
#include "core/drivers/processor/program.hpp"
#include "core/memory/allocation.hpp"
#include "core/bytecode/accumulator.hpp"
#include "core/bytecode/instruction.hpp"
#include "core/bytecode/precision.hpp"
#include "core/bytecode/fold.hpp"
#include "core/bytecode/code.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(SCI_ACCELERATE)
#include <vecLib/cblas_new.h>
#endif

namespace sci::core::drivers::processor {

    constexpr size block = Plan::block;

    // Per-thread buffer for a chunk's constants and temporaries, one block each.
    static thread_local memory::Allocation workspace;

    // Runs the first `steps` instructions over [first, last) block by block, then hands the
    // value table and the block length to `visit`.
    template <typename type, typename visitor>
    static void sweep(const kernels::Routines<type>& routines, const Chunk& chunk, type* output, const size first, const size last,
               const uint8 steps, visitor& visit) {
        // Indexed by value id; only the ids the chunk can reference are filled in.
        type* values[256];
        values[slot::none] = nullptr;
        values[slot::output] = output;

        const size literals = chunk.literals;
        auto* space = static_cast<type*>(workspace.reserve((literals + chunk.temporaries) * block * sizeof(type)));
        for (size index = 0; index < literals; ++index) {
            type* constant = space + index * block;
            std::fill_n(constant, block, static_cast<type>(chunk.constants[index]));
            values[slot::literal | index] = constant;
        }
        for (size index = 0; index < chunk.temporaries; ++index) {
            values[slot::temporary | index] = space + (literals + index) * block;
        }

        for (size start = first; start < last; start += block) {
            const size count = std::min(block, last - start);
            for (size index = 0; index < chunk.arguments; ++index) {
                values[slot::input | index] = const_cast<type*>(static_cast<const type*>(chunk.inputs[index])) + start;
            }
            if (output) {
                values[slot::output] = output + start;
            }
            for (uint8 index = 0; index < steps; ++index) {
                const Instruction& step = chunk.code[index];
                const type* sources[3] = {values[step.sources[0]], values[step.sources[1]], values[step.sources[2]]};
                routines.kernels[static_cast<size>(step.code)](sources, values[step.target], count);
            }
            visit(values, count);
        }
    }

    // A compiled program covers whole registers of each part; the kernels finish the elements left
    // at the end.
    template <typename type>
    static void evaluate(const kernels::Routines<type>& routines, const Chunk& chunk, type* output, const size count,
                         const Program<type>& program) {
        if (count == 0) return;
        const Plan plan(chunk, count);
        auto work = [&](const size part) {
            const auto [first, last] = plan.range(part);
            size start = first;
            if (program) {
                const size covered = program.cover(last - first);
                if (covered > 0) program.run(output + first, first, covered);
                start = first + covered;
            }
            if (start < last) {
                auto ignore = [](type* const*, size) noexcept {};
                sweep(routines, chunk, output, start, last, chunk.length, ignore);
            }
        };
        if (plan.parts == 1) {
            work(0);
        } else {
            Pool::global().run(plan.parts, work, plan.seats);
        }
    }

    template <typename type>
    static double accumulate(const kernels::Routines<type>& routines, const Chunk& chunk, const Fold fold, const size count,
                             const Program<type>& program) {
        if (count == 0) {
            return fold == Fold::sum ? 0.0 : std::numeric_limits<double>::quiet_NaN();
        }

        // A contracted sum skips the final multiply and takes the dot product of its operands.
        const Instruction* product = chunk.length > 0 ? &chunk.code[chunk.length - 1] : nullptr;
        const bool contract = chunk.contracted(fold);
        const uint8 steps = contract ? chunk.length - 1 : chunk.length;
        const auto reduction = routines.folds[static_cast<size>(fold)];

        const Plan plan(chunk, count);

        // Float sums are folded per block in the native type, then accumulated in double. Compiled
        // chunks fill one block at a time and fold it with the same routine, so the result is the
        // same either way.
        auto segment = [&](const size part) {
            const auto [first, end] = plan.range(part);
            Accumulator accumulator(fold);
            if (program && !contract) {
                alignas(64) type results[block];
                for (size start = first; start < end; start += block) {
                    const size length = std::min(block, end - start);
                    const size covered = program.cover(length);
                    if (covered > 0) program.run(results, start, covered);
                    if (covered < length) {
                        auto gather = [&](type* const* values, const size rest) { std::copy_n(values[chunk.result], rest, results + covered); };
                        sweep(routines, chunk, static_cast<type*>(nullptr), start + covered, start + length, steps, gather);
                    }
                    accumulator.add(static_cast<double>(reduction(results, length)));
                }
                return accumulator.result();
            }
            auto visit = [&](type* const* values, const size length) {
                accumulator.add(static_cast<double>(contract
                    ? routines.dot(values[product->sources[0]], values[product->sources[1]], length)
                    : reduction(values[chunk.result], length)));
            };
            sweep(routines, chunk, static_cast<type*>(nullptr), first, end, steps, visit);
            return accumulator.result();
        };

        if (plan.parts == 1) {
            return segment(0);
        }

        std::vector<double> partials(plan.parts);
        Pool::global().run(plan.parts, [&](const size part) { partials[part] = segment(part); }, plan.seats);

        Accumulator accumulator(fold);
        for (const double partial : partials) {
            accumulator.add(partial);
        }
        return accumulator.result();
    }

    // Register-blocked matrix product, used where no system BLAS is available: each step keeps a
    // 4 x line block of the output in registers (line = one cache line of columns) and streams a
    // column panel of `right` that stays in L1 while a band of rows passes over it. Plain loops;
    // the compiler maps them onto the vector unit.
    template <typename type>
    static void multiply(const type* left, const type* right, type* output, const size rows, const size columns, const size depth) {
        constexpr size height = 4;
        constexpr size line = 64 / sizeof(type);
        constexpr size band = 32;

        const size bands = (rows + band - 1) / band;
        auto work = [&](const size task) {
            const size top = task * band;
            const size bottom = std::min(rows, top + band);
            for (size column = 0; column < columns; column += line) {
                const size span = std::min(line, columns - column);
                for (size row = top; row < bottom; row += height) {
                    const size tall = std::min(height, bottom - row);
                    if (tall == height && span == line) {
                        type sums[height][line]{};
                        for (size inner = 0; inner < depth; ++inner) {
                            const type* panel = right + inner * columns + column;
                            for (size down = 0; down < height; ++down) {
                                const type scale = left[(row + down) * depth + inner];
                                for (size across = 0; across < line; ++across) {
                                    sums[down][across] += scale * panel[across];
                                }
                            }
                        }
                        for (size down = 0; down < height; ++down) {
                            for (size across = 0; across < line; ++across) {
                                output[(row + down) * columns + column + across] = sums[down][across];
                            }
                        }
                    } else {
                        for (size down = 0; down < tall; ++down) {
                            for (size across = 0; across < span; ++across) {
                                type sum{};
                                for (size inner = 0; inner < depth; ++inner) {
                                    sum += left[(row + down) * depth + inner] * right[inner * columns + column + across];
                                }
                                output[(row + down) * columns + column + across] = sum;
                            }
                        }
                    }
                }
            }
        };

        if (bands > 1 && rows * columns * depth >= (size{1} << 18)) {
            Pool::global().run(bands, work);
        } else {
            for (size task = 0; task < bands; ++task) work(task);
        }
    }

    Dispatch::Dispatch(const Context context) noexcept {
        if (context.avx512) table = kernels::avx512();
        if (!table && context.avx2) table = kernels::avx2();
        if (!table && context.neon) table = kernels::neon();
        if (!table) table = kernels::fallback();
    }

    Dispatch::Dispatch(const kernels::Table* table) noexcept
        : table(table ? table : kernels::fallback()) {}

    const Dispatch& Dispatch::global() noexcept {
        static const Dispatch instance = [] {
            if (const char* name = std::getenv("SCI_ISA")) {
                if (const kernels::Table* chosen = select(name)) return Dispatch(chosen);
            }
            return Dispatch(Context::probe());
        }();
        return instance;
    }

    const kernels::Table* Dispatch::select(const char* name) noexcept {
        if (!name) return nullptr;
        const Context context = Context::probe();
        if (std::strcmp(name, "fallback") == 0) return kernels::fallback();
        if (std::strcmp(name, "neon") == 0 && context.neon) return kernels::neon();
        if (std::strcmp(name, "avx2") == 0 && context.avx2) return kernels::avx2();
        if (std::strcmp(name, "avx512") == 0 && context.avx512) return kernels::avx512();
        return nullptr;
    }

    const char* Dispatch::name() const noexcept {
        return table->name;
    }

    void Dispatch::execute(const Chunk& chunk, void* output, const size count) const {
        const Compiler::Function function = Compiler::global().compile(chunk, *table, count);
        const size width = Compiler::width(*table, chunk.precision);
        if (chunk.precision == Precision::single) {
            evaluate(table->single, chunk, static_cast<float*>(output), count, Program<float>(function, chunk, width));
        } else {
            evaluate(table->wide, chunk, static_cast<double*>(output), count, Program<double>(function, chunk, width));
        }
    }

    double Dispatch::reduce(const Chunk& chunk, const Fold fold, const size count) const {
        // A contracted sum is already a dot product over the multiply's operands.
        const Compiler::Function function = chunk.contracted(fold) ? nullptr : Compiler::global().compile(chunk, *table, count);
        const size width = Compiler::width(*table, chunk.precision);
        if (chunk.precision == Precision::single) {
            return accumulate(table->single, chunk, fold, count, Program<float>(function, chunk, width));
        }
        return accumulate(table->wide, chunk, fold, count, Program<double>(function, chunk, width));
    }

    void Dispatch::product(const Precision precision, const void* left, const void* right, void* output, const size rows, const size columns, const size depth) {
        const size bytes = precision == Precision::single ? sizeof(float) : sizeof(double);
        if (depth == 0) {
            std::memset(output, 0, rows * columns * bytes);
            return;
        }

        #if defined(SCI_ACCELERATE)
            // Apple's BLAS (the AMX matrix units on M-series): what NumPy uses on macOS too.
            if (rows <= INT_MAX && columns <= INT_MAX && depth <= INT_MAX) {
                const int height = static_cast<int>(rows);
                const int width = static_cast<int>(columns);
                const int inner = static_cast<int>(depth);
                if (precision == Precision::single) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, height, width, inner, 1.0f,
                                static_cast<const float*>(left), inner, static_cast<const float*>(right), width, 0.0f, static_cast<float*>(output), width);
                } else {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, height, width, inner, 1.0,
                                static_cast<const double*>(left), inner, static_cast<const double*>(right), width, 0.0, static_cast<double*>(output), width);
                }
                return;
            }
        #endif

        if (precision == Precision::single) {
            multiply(static_cast<const float*>(left), static_cast<const float*>(right), static_cast<float*>(output), rows, columns, depth);
        } else {
            multiply(static_cast<const double*>(left), static_cast<const double*>(right), static_cast<double*>(output), rows, columns, depth);
        }
    }

}
