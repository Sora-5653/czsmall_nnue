// SPDX-License-Identifier: MIT
#pragma once

#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

namespace tetra_test {

struct TestCase {
    const char* name;
    void (*fn)();
};

std::vector<TestCase>& registry();
extern int g_checks;
extern int g_failures;
extern const char* g_current;

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back(TestCase{name, fn}); }
};

// Best-effort stringification: numeric types print their value, anything else
// (std::string, enums, structs) falls back to a placeholder so CHECK_EQ works
// uniformly without every call site needing an overload.
template <typename T>
inline std::string describe(const T& v) {
    if constexpr (std::is_arithmetic_v<T>) return std::to_string(v);
    else if constexpr (std::is_enum_v<T>)
        return std::to_string(static_cast<long long>(v));
    else if constexpr (std::is_convertible_v<T, std::string>)
        return std::string(v);
    else
        return "<value>";
}

inline void report_failure(const char* file, int line, const std::string& msg) {
    ++g_failures;
    std::printf("       %s:%d: %s\n", file, line, msg.c_str());
}

}  // namespace tetra_test

#define TEST(name)                                                        \
    static void name();                                                   \
    static ::tetra_test::Registrar reg_##name(#name, name);               \
    static void name()

#define CHECK(cond)                                                       \
    do {                                                                  \
        ++::tetra_test::g_checks;                                         \
        if (!(cond))                                                      \
            ::tetra_test::report_failure(__FILE__, __LINE__,              \
                                         "CHECK failed: " #cond);         \
    } while (0)

#define CHECK_EQ(a, b)                                                    \
    do {                                                                  \
        ++::tetra_test::g_checks;                                         \
        auto _va = (a);                                                   \
        auto _vb = (b);                                                   \
        if (!(_va == _vb)) {                                              \
            ::tetra_test::report_failure(                                 \
                __FILE__, __LINE__,                                       \
                std::string("CHECK_EQ failed: " #a " == " #b " (got ") +  \
                    ::tetra_test::describe(_va) + " vs " +                \
                    ::tetra_test::describe(_vb) + ")");                   \
        }                                                                 \
    } while (0)

#define CHECK_MSG(cond, msg)                                              \
    do {                                                                  \
        ++::tetra_test::g_checks;                                         \
        if (!(cond)) ::tetra_test::report_failure(__FILE__, __LINE__, (msg)); \
    } while (0)