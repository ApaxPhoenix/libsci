#include "core/drivers/cpu/dispatch.hpp"
#include "core/arch/alignment.hpp"
#include "core/drivers/cpu/context.hpp"
#include "core/drivers/cpu/jit.hpp"
#include "core/drivers/cpu/kernels/avx2.hpp"
#include "core/drivers/cpu/kernels/avx512.hpp"
#include "core/drivers/cpu/kernels/fallback.hpp"
#include "core/drivers/cpu/kernels/neon.hpp"
#include "core/drivers/cpu/kernels/routines.hpp"
#include "core/drivers/cpu/kernels/table.hpp"
#include "core/drivers/cpu/pool.hpp"
#include "core/bytecode/instruction.hpp"
#include "core/bytecode/dtype.hpp"
#include "core/bytecode/fold.hpp"
#include "core/bytecode/code.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#if defined(SCI_ACCELERATE)
#include <vecLib/cblas_new.h>
#endif

namespace sci::core::drivers::cpu {

    // Elements per block. Every temporary of an expression is one block, so its whole
    // working set stays in L1 while it is evaluated (1024 measured best on an M1).
    constexpr size block = 1024;

    // Weighted element count below which forking costs more than it saves.
    constexpr size threshold = size{1} << 17;

    // Chunks this cheap per element only stream memory...
    constexpr uint32 streaming = 16;

    // ...and once they touch more than the caches hold, DRAM is the limit: extra slow cores on a
    // hybrid chip only add contention (measured on an M1: 4 threads beat 8 by about 15%).
    constexpr size spill = size{32} << 20;

    // Per-thread buffer for a chunk's constants and temporaries, one block each.
    struct Scratch {
        void* data{nullptr};
        size bytes{0};

        ~Scratch() {
            if (data) ::operator delete(data, std::align_val_t{arch::cpu});
        }

        void* reserve(const size amount) {
            if (amount > bytes) {
                if (data) ::operator delete(data, std::align_val_t{arch::cpu});
                data = ::operator new(amount, std::align_val_t{arch::cpu});
                bytes = amount;
            }
            return data;
        }
    };

    static thread_local Scratch workspace;

    // Folds two partial results; min and max let NaN win, as numpy does.
    static double combine(const Fold kind, const double left, const double right) noexcept {
        if (kind == Fold::sum) return left + right;
        if (std::isnan(left) || std::isnan(right)) return left + right;
        return kind == Fold::min ? std::min(left, right) : std::max(left, right);
    }

    // Starting value of a reduction: combining with it changes nothing.
    static double identity(const Fold kind) noexcept {
        constexpr double infinity = std::numeric_limits<double>::infinity();
        return kind == Fold::sum ? 0.0 : (kind == Fold::min ? infinity : -infinity);
    }

    // How the work is split: parts of whole blocks, and how many threads may take part.
    struct Plan {
        size parts;
        size seats;
    };

    static Plan partition(const Chunk& chunk, const size count) {
        const size blocks = (count + block - 1) / block;
        const size cost = chunk.cost();
        if (blocks < 2 || count * cost < threshold) return {.parts = 1, .seats = 1};
        const size width = Pool::global().width();
        if (width < 2) return {.parts = 1, .seats = 1};

        const size element = chunk.type == Dtype::float32 ? sizeof(float) : sizeof(double);
        const size touched = count * element * (chunk.arguments + 1);
        const size seats = cost <= streaming && touched >= spill ? Context::performance() : 0;

        // Several parts per thread so faster cores pick up the slack.
        return {.parts = std::min(blocks, width * 8), .seats = seats};
    }

    struct Range {
        size first;
        size last;
    };

    static Range bounds(const size part, const size parts, const size count) noexcept {
        const size blocks = (count + block - 1) / block;
        return {.first = blocks * part / parts * block, .last = std::min(blocks * (part + 1) / parts * block, count)};
    }

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

    // sum(... * ...): the final multiply is skipped and its operands contracted directly (a dot
    // product), so the product never round-trips through a temporary block.
    static bool contracted(const Chunk& chunk, const Fold kind) noexcept {
        if (kind != Fold::sum || chunk.length == 0) return false;
        const Instruction& last = chunk.code[chunk.length - 1];
        return last.code == Code::mul && last.target == chunk.result;
    }

    // The chunk's constants in the element type, as compiled chunks read them.
    template <typename type>
    static void convert(const Chunk& chunk, type* constants) noexcept {
        for (size index = 0; index < chunk.literals; ++index) {
            constants[index] = static_cast<type>(chunk.constants[index]);
        }
    }

    // Runs a compiled chunk over elements [first, first + count) of its inputs, writing to `output`
    // (where element `first` goes). count is a multiple of the compiled code's width.
    template <typename type>
    static void launch(const Jit::Function function, const Chunk& chunk, const type* constants, type* output, const size first, const size count) {
        const void* inputs[Chunk::capacity];
        for (size index = 0; index < chunk.arguments; ++index) {
            inputs[index] = static_cast<const type*>(chunk.inputs[index]) + first;
        }
        function(inputs, constants, output, count);
    }

    // `function`, when not null, is the chunk compiled by the Jit: it covers whole registers of
    // `width` elements, and the kernels finish the elements left at the end.
    template <typename type>
    static void evaluate(const kernels::Routines<type>& routines, const Chunk& chunk, type* output, const size count,
                         const Jit::Function function, const size width) {
        if (count == 0) return;
        type constants[Chunk::capacity];
        if (function) convert(chunk, constants);
        const Plan plan = partition(chunk, count);
        auto work = [&](const size part) {
            const auto [first, last] = bounds(part, plan.parts, count);
            size start = first;
            if (function) {
                const size covered = (last - first) / width * width;
                if (covered > 0) launch(function, chunk, constants, output + first, first, covered);
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
    static double accumulate(const kernels::Routines<type>& routines, const Chunk& chunk, const Fold kind, const size count,
                             const Jit::Function function, const size width) {
        if (count == 0) {
            return kind == Fold::sum ? 0.0 : std::numeric_limits<double>::quiet_NaN();
        }

        const Instruction* last = chunk.length > 0 ? &chunk.code[chunk.length - 1] : nullptr;
        const bool contract = contracted(chunk, kind);
        const uint8 steps = contract ? chunk.length - 1 : chunk.length;
        const auto fold = routines.folds[static_cast<size>(kind)];
        type constants[Chunk::capacity];
        if (function) convert(chunk, constants);

        const Plan plan = partition(chunk, count);

        // Float sums are folded per block in the native type, then accumulated in double. Compiled
        // chunks fill one block at a time and fold it with the same routine, so the result is the
        // same either way.
        auto segment = [&](const size part) {
            const auto [first, end] = bounds(part, plan.parts, count);
            double result = identity(kind);
            if (function && !contract) {
                alignas(64) type results[block];
                for (size start = first; start < end; start += block) {
                    const size length = std::min(block, end - start);
                    const size covered = length / width * width;
                    if (covered > 0) launch(function, chunk, constants, results, start, covered);
                    if (covered < length) {
                        auto gather = [&](type* const* values, const size rest) { std::copy_n(values[chunk.result], rest, results + covered); };
                        sweep(routines, chunk, static_cast<type*>(nullptr), start + covered, start + length, steps, gather);
                    }
                    result = combine(kind, result, static_cast<double>(fold(results, length)));
                }
                return result;
            }
            auto visit = [&](type* const* values, const size length) {
                const auto part = static_cast<double>(contract
                    ? routines.dot(values[last->sources[0]], values[last->sources[1]], length)
                    : fold(values[chunk.result], length));
                result = combine(kind, result, part);
            };
            sweep(routines, chunk, static_cast<type*>(nullptr), first, end, steps, visit);
            return result;
        };

        if (plan.parts == 1) {
            return segment(0);
        }

        std::vector<double> partials(plan.parts);
        Pool::global().run(plan.parts, [&](const size part) { partials[part] = segment(part); }, plan.seats);

        double result = identity(kind);
        for (const double part : partials) {
            result = combine(kind, result, part);
        }
        return result;
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
                            for (size i = 0; i < height; ++i) {
                                const type scale = left[(row + i) * depth + inner];
                                for (size j = 0; j < line; ++j) {
                                    sums[i][j] += scale * panel[j];
                                }
                            }
                        }
                        for (size i = 0; i < height; ++i) {
                            for (size j = 0; j < line; ++j) {
                                output[(row + i) * columns + column + j] = sums[i][j];
                            }
                        }
                    } else {
                        for (size i = 0; i < tall; ++i) {
                            for (size j = 0; j < span; ++j) {
                                type sum{};
                                for (size inner = 0; inner < depth; ++inner) {
                                    sum += left[(row + i) * depth + inner] * right[inner * columns + column + j];
                                }
                                output[(row + i) * columns + column + j] = sum;
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
        const Jit::Function function = Jit::global().compile(chunk, *table, count);
        const size width = Jit::width(*table, chunk.type);
        if (chunk.type == Dtype::float32) {
            evaluate(table->float32, chunk, static_cast<float*>(output), count, function, width);
        } else {
            evaluate(table->float64, chunk, static_cast<double*>(output), count, function, width);
        }
    }

    double Dispatch::reduce(const Chunk& chunk, const Fold kind, const size count) const {
        // A contracted sum is already a dot product over the multiply's operands.
        const Jit::Function function = contracted(chunk, kind) ? nullptr : Jit::global().compile(chunk, *table, count);
        const size width = Jit::width(*table, chunk.type);
        if (chunk.type == Dtype::float32) {
            return accumulate(table->float32, chunk, kind, count, function, width);
        }
        return accumulate(table->float64, chunk, kind, count, function, width);
    }

    void Dispatch::matmul(const Dtype type, const void* left, const void* right, void* output, const size rows, const size columns, const size depth) {
        const size width = type == Dtype::float32 ? sizeof(float) : sizeof(double);
        if (depth == 0) {
            std::memset(output, 0, rows * columns * width);
            return;
        }

        #if defined(SCI_ACCELERATE)
            // Apple's BLAS (the AMX matrix units on M-series): what NumPy uses on macOS too.
            if (rows <= INT_MAX && columns <= INT_MAX && depth <= INT_MAX) {
                const int m = static_cast<int>(rows);
                const int n = static_cast<int>(columns);
                const int k = static_cast<int>(depth);
                if (type == Dtype::float32) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                                static_cast<const float*>(left), k, static_cast<const float*>(right), n, 0.0f, static_cast<float*>(output), n);
                } else {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                                static_cast<const double*>(left), k, static_cast<const double*>(right), n, 0.0, static_cast<double*>(output), n);
                }
                return;
            }
        #endif

        if (type == Dtype::float32) {
            multiply(static_cast<const float*>(left), static_cast<const float*>(right), static_cast<float*>(output), rows, columns, depth);
        } else {
            multiply(static_cast<const double*>(left), static_cast<const double*>(right), static_cast<double*>(output), rows, columns, depth);
        }
    }

}
