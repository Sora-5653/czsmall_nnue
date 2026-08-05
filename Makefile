# TetraFormer / TetraZero -- M0 (rule core) + M1 (movegen/tokenizer) build.
#
# Deliberately dependency-free: a stock g++/clang++ with C++23 is all that is
# required. The vendored Cobra movegen backend requires C++23.

CXX      ?= g++
CXXSTD   ?= -std=c++23
WARN     ?= -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
OPT      ?= -O2
DEPFLAGS := -MMD -MP
CXXFLAGS ?= $(CXXSTD) $(WARN) $(OPT) -Iinclude $(DEPFLAGS)
LDFLAGS  ?=

BUILD := build
SRC   := $(wildcard src/*.cpp)
OBJ   := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRC))

TEST_SRC := $(wildcard tests/*.cpp)
TEST_BIN := $(BUILD)/tetra_tests

TOOL_SRC := $(wildcard tools/*.cpp)
TOOL_BIN := $(patsubst tools/%.cpp,$(BUILD)/%,$(TOOL_SRC))
TOOL_HDR := $(wildcard include/tetra/*.hpp)

.PHONY: all test tools clean fmt

all: $(TEST_BIN) tools

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

TEST_OBJ := $(patsubst tests/%.cpp,$(BUILD)/tests_%.o,$(TEST_SRC))

$(BUILD)/tests_%.o: tests/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Itests -c $< -o $@

$(TEST_BIN): $(TEST_OBJ) $(OBJ) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(TEST_OBJ) $(OBJ) -o $@ $(LDFLAGS)

$(BUILD)/%: tools/%.cpp $(OBJ) $(TOOL_HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(OBJ) -o $@ $(LDFLAGS)

tools: $(TOOL_BIN)

test: $(TEST_BIN)
	@./$(TEST_BIN)

# Stricter pass used in CI: sanitizers + debug assertions.
.PHONY: test-asan
test-asan:
	@mkdir -p $(BUILD)
	$(CXX) $(CXXSTD) $(WARN) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
	    -Iinclude $(TEST_SRC) $(SRC) -o $(BUILD)/tetra_tests_asan
	@./$(BUILD)/tetra_tests_asan

clean:
	@rm -rf $(BUILD)

# Header dependency tracking, so editing a header rebuilds its dependents.
-include $(wildcard $(BUILD)/*.d)
