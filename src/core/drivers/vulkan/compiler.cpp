#include "core/drivers/vulkan/compiler.hpp"
#include "core/bytecode/chunk.hpp"
#include "core/bytecode/precision.hpp"
#include "core/bytecode/instruction.hpp"
#include "core/bytecode/code.hpp"
#include "core/drivers/vulkan/context.hpp"
#include "core/drivers/vulkan/kernel.hpp"
#include "core/drivers/vulkan/pipeline.hpp"
#include "core/types.hpp"
#include "shaders/sources.hpp"

#include <slang-com-ptr.h>
#include <slang.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <print>
#include <span>
#include <string>
#include <vector>

namespace sci::core::drivers::vulkan {

    // Distinct chunks compiled at most; later ones keep running on the interpreter.
    constexpr size limit = 1024;

    // Slang's name for each one-operand function, by code (absolute through hyperbolic::arctangent).
    constexpr const char* functions[] = {
        nullptr, nullptr, "abs", "sqrt", "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    };

    std::string Compiler::generate(const Chunk& chunk) {
        std::string names[256];
        std::string body;
        for (uint8 index = 0; index < chunk.arguments; ++index) {
            names[slot::input | index] = std::format("input{}", index);
            body += std::format("    const float4 input{0} = load(chunk, {0}, index);\n", index);
        }
        for (uint8 index = 0; index < chunk.literals; ++index) {
            names[slot::literal | index] = std::format("literal{}", index);
            body += std::format("    const float4 literal{0} = float4(chunk.constants[{0}]);\n", index);
        }
        for (uint8 step = 0; step < chunk.length; ++step) {
            const Instruction& instruction = chunk.code[step];
            const std::string& first = names[instruction.sources[0]];
            const std::string& second = names[instruction.sources[1]];
            const std::string& third = names[instruction.sources[2]];
            std::string expression;
            if (instruction.code == Code::copy) {
                expression = first;
            } else if (instruction.code == Code::negate) {
                expression = std::format("-{}", first);
            } else if (instruction.code < Code::add) {
                expression = std::format("{}({})", functions[static_cast<size>(instruction.code)], first);
            } else {
                switch (instruction.code) {
                    case Code::add: expression = std::format("{} + {}", first, second); break;
                    case Code::subtract: expression = std::format("{} - {}", first, second); break;
                    case Code::multiply: expression = std::format("{} * {}", first, second); break;
                    case Code::divide: expression = std::format("{} / {}", first, second); break;
                    case Code::minimum: expression = std::format("lesser({}, {})", first, second); break;
                    case Code::maximum: expression = std::format("greater({}, {})", first, second); break;
                    default: expression = std::format("fma({}, {}, {})", first, second, third); break;
                }
            }
            body += std::format("    const float4 value{} = {};\n", step, expression);
            if (instruction.target == slot::output) {
                body += std::format("    store(index, live, value{});\n", step);
            }
            names[instruction.target] = std::format("value{}", step);
        }
        body += std::format("    return {};\n", names[chunk.result]);
        return std::format("{}\nfloat4 run(Chunk* chunk, uint4 index, bool4 live) {{\n{}}}\n\n{}",
                           shaders::sources::layout, body, shaders::sources::entries);
    }

    std::vector<std::vector<uint32>> Compiler::compile(slang::IGlobalSession* global, const std::string& source, std::string& diagnostics) {
        slang::TargetDesc target{};
        target.format = SLANG_SPIRV;
        slang::CompilerOptionEntry options[2]{};
        options[0].name = slang::CompilerOptionName::VulkanUseEntryPointName;
        options[0].value.intValue0 = 1;
        options[1].name = slang::CompilerOptionName::Optimization;
        options[1].value.intValue0 = SLANG_OPTIMIZATION_LEVEL_MAXIMAL;
        slang::SessionDesc description{};
        description.targets = &target;
        description.targetCount = 1;
        description.compilerOptionEntries = options;
        description.compilerOptionEntryCount = 2;

        Slang::ComPtr<slang::ISession> session;
        if (SLANG_FAILED(global->createSession(description, session.writeRef()))) return {};

        Slang::ComPtr<slang::IBlob> messages;
        const auto report = [&] {
            if (messages) diagnostics.append(static_cast<const char*>(messages->getBufferPointer()), messages->getBufferSize());
        };
        slang::IModule* module = session->loadModuleFromSourceString("chunk", "chunk.slang", source.c_str(), messages.writeRef());
        report();
        if (!module) return {};

        Slang::ComPtr<slang::IEntryPoint> evaluate;
        Slang::ComPtr<slang::IEntryPoint> reduce;
        if (SLANG_FAILED(module->findEntryPointByName("evaluate", evaluate.writeRef())) ||
            SLANG_FAILED(module->findEntryPointByName("reduce", reduce.writeRef()))) {
            return {};
        }
        slang::IComponentType* parts[] = {module, evaluate.get(), reduce.get()};
        Slang::ComPtr<slang::IComponentType> program;
        if (SLANG_FAILED(session->createCompositeComponentType(parts, 3, program.writeRef(), messages.writeRef()))) {
            report();
            return {};
        }
        Slang::ComPtr<slang::IComponentType> linked;
        if (SLANG_FAILED(program->link(linked.writeRef(), messages.writeRef()))) {
            report();
            return {};
        }

        std::vector<std::vector<uint32>> modules;
        for (SlangInt entry = 0; entry < 2; ++entry) {
            Slang::ComPtr<slang::IBlob> code;
            if (SLANG_FAILED(linked->getEntryPointCode(entry, 0, code.writeRef(), messages.writeRef()))) {
                report();
                return {};
            }
            const auto* words = static_cast<const uint32*>(code->getBufferPointer());
            modules.emplace_back(words, words + code->getBufferSize() / sizeof(uint32));
        }
        return modules;
    }

    Compiler::Compiler(const Context& context, const uint32 bytes)
        : context(context), bytes(bytes) {
        if (const char* setting = std::getenv("SCI_JIT")) {
            enabled = std::strcmp(setting, "0") != 0;
            tracing = std::strcmp(setting, "trace") == 0;
        }
    }

    Compiler::~Compiler() {
        {
            std::scoped_lock lock(guard);
            stopping = true;
        }
        signal.notify_all();
        if (worker.joinable()) worker.join();
    }

    const Kernel* Compiler::find(const Chunk& chunk) {
        if (!enabled || !chunk.valid() || chunk.length == 0 || chunk.precision != Precision::single) return nullptr;

        std::string signature = chunk.signature();
        std::scoped_lock lock(guard);
        if (const auto found = kernels.find(signature); found != kernels.end()) return found->second.get();
        if (kernels.size() >= limit) return nullptr;

        jobs.emplace_back(signature, generate(chunk));
        kernels.emplace(std::move(signature), nullptr);
        if (!worker.joinable()) worker = std::thread([this] { loop(); });
        signal.notify_one();
        return nullptr;
    }

    void Compiler::loop() {
        // Loads Slang's core module, which takes a while, so it happens here rather than in find().
        Slang::ComPtr<slang::IGlobalSession> global;
        if (SLANG_FAILED(slang_createGlobalSession(SLANG_API_VERSION, global.writeRef()))) {
            enabled = false;
            if (tracing) std::println(stderr, "[sci] jit: Slang could not start; GPU chunks stay on the interpreter");
            return;
        }

        while (true) {
            std::pair<std::string, std::string> job;
            {
                std::unique_lock lock(guard);
                signal.wait(lock, [this] { return stopping || !jobs.empty(); });
                if (stopping) return;
                job = std::move(jobs.front());
                jobs.pop_front();
            }

            std::string diagnostics;
            const std::vector<std::vector<uint32>> modules = compile(global.get(), job.second, diagnostics);
            std::unique_ptr<Kernel> kernel;
            if (modules.size() == 2) {
                kernel = std::make_unique<Kernel>();
                kernel->evaluate = std::make_unique<Pipeline>(context, std::span<const uint32>(modules[0]), "evaluate", bytes);
                kernel->reduce = std::make_unique<Pipeline>(context, std::span<const uint32>(modules[1]), "reduce", bytes);
                if (!kernel->evaluate->valid() || !kernel->reduce->valid()) kernel.reset();
            }
            if (tracing) {
                std::println(stderr, "[sci] jit: {} a GPU kernel{}{}", kernel ? "compiled" : "could not compile",
                             diagnostics.empty() ? "" : "\n", diagnostics);
            }

            std::scoped_lock lock(guard);
            kernels[job.first] = std::move(kernel);
        }
    }

}
