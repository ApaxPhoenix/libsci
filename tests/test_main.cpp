#include <cassert>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
#define WEXITSTATUS(status) (status)
#else
#include <sys/wait.h>
#endif

namespace tests {
    int run() {
        const auto status = std::system((std::string("\"") + BINARY + "\" > " + NULL_DEVICE + " 2>&1").c_str());
        return WEXITSTATUS(status);
    }
}

int main() {
    assert((tests::run() == 0) && "the demonstration's own checks pass, and it runs to the end");
    return 0;
}
