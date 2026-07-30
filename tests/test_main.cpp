// SPDX-License-Identifier: MIT
// Minimal dependency-free test harness (spec 18).
#include "test_util.hpp"

#include <cstdio>

namespace tetra_test {

std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

int g_checks = 0;
int g_failures = 0;
const char* g_current = "";

}  // namespace tetra_test

int main(int argc, char** argv) {
    using namespace tetra_test;
    const char* filter = (argc > 1) ? argv[1] : nullptr;

    int run = 0, failed = 0;
    for (auto& tc : registry()) {
        if (filter && std::string(tc.name).find(filter) == std::string::npos) continue;
        g_current = tc.name;
        const int before = g_failures;
        ++run;
        tc.fn();
        if (g_failures > before) {
            ++failed;
            std::printf("  \033[31mFAIL\033[0m %s\n", tc.name);
        } else {
            std::printf("  \033[32mok\033[0m   %s\n", tc.name);
        }
    }

    std::printf("\n%d tests, %d assertions, %d failed\n", run, g_checks, failed);
    return failed == 0 ? 0 : 1;
}
