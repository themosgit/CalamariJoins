cmake_minimum_required(VERSION 3.16)

project(SigmodContest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

Include(FetchContent)
FetchContent_Declare( Catch2 URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.8.0.tar.gz)
FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
    abseil
    URL https://github.com/abseil/abseil-cpp/releases/download/20240722.1/abseil-cpp-20240722.1.tar.gz
)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL ON)
FetchContent_MakeAvailable(abseil)
FetchContent_Declare(
    re2
    URL https://github.com/google/re2/releases/download/2024-07-02/re2-2024-07-02.tar.gz
)


FetchContent_MakeAvailable(re2)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_MakeAvailable(json)

FetchContent_Declare(
    sql-parser
    URL https://github.com/a858438680/sql-parser/archive/refs/tags/win-port-2.tar.gz
)
set(HSQL_ENABLE_WERROR OFF)
FetchContent_MakeAvailable(sql-parser)

FetchContent_Declare(
    range-v3
    URL https://github.com/ericniebler/range-v3/archive/refs/tags/0.12.0.tar.gz
)

FetchContent_MakeAvailable(range-v3)

FetchContent_Declare(
    fmtlib
    URL https://github.com/fmtlib/fmt/releases/download/11.1.3/fmt-11.1.3.zip
)

FetchContent_MakeAvailable(fmtlib)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|powerpc|ppc64|ppc64le")
    message("Disabling jemalloc extension of DuckDB on Power.")
    set(SKIP_EXTENSIONS jemalloc)
endif()

# Detect Xeon E5-2680 v3 CPU for benchmark VM hardware configuration
# Requires both: correct CPU AND at least 32GB RAM (benchmark VM has 64GB, CI has 4GB)
set(IS_BENCHMARK_VM_HARDWARE OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" CPUINFO_CONTENT)
    if(CPUINFO_CONTENT MATCHES "E5-2680 v3")
        # Check available memory to distinguish benchmark VM from CI VM
        if(EXISTS "/proc/meminfo")
            file(READ "/proc/meminfo" MEMINFO_CONTENT)
            string(REGEX MATCH "MemTotal:[ \t]+([0-9]+)" MEM_MATCH "${MEMINFO_CONTENT}")
            if(MEM_MATCH)
                set(MEM_TOTAL_KB "${CMAKE_MATCH_1}")
                math(EXPR MEM_TOTAL_GB "${MEM_TOTAL_KB} / 1024 / 1024")
                if(MEM_TOTAL_GB GREATER_EQUAL 32)
                    message(STATUS "Detected Intel Xeon E5-2680 v3 CPU with ${MEM_TOTAL_GB}GB RAM - using benchmark VM hardware configuration")
                    add_compile_definitions(SPC__USE_BENCHMARKVM_HARDWARE)
                    set(IS_BENCHMARK_VM_HARDWARE ON)
                else()
                    message(STATUS "Detected Intel Xeon E5-2680 v3 CPU but only ${MEM_TOTAL_GB}GB RAM (need >=32GB) - using generic hardware configuration")
                endif()
            endif()
        endif()
    endif()
endif()

# Include all sources from /src directory. CONFIGURE_DEPENDS can be unreliable.
# Try re-running cmake in case changes are not recognized.
file(GLOB ALL_SRC
    CONFIGURE_DEPENDS
    "src/*.cpp"
)

set(MANOLATES_SRC ${ALL_SRC})

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    add_compile_options(-O3 -mcpu=apple-m1 -flto)
else()
    add_compile_options(-O3 -march=skylake -m64 -mcrc32 -fpermissive -flto)
endif()

add_link_options(-flto)

add_executable(unit_tests ${MANOLATES_SRC} tests/unit_tests.cpp)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_executable(faster ${MANOLATES_SRC} tests/read_sql.cpp)
    target_compile_definitions(faster PRIVATE)
    target_link_libraries(faster PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
    target_include_directories(faster PRIVATE include)

    if(IS_BENCHMARK_VM_HARDWARE)
        add_executable(leaderboard ${MANOLATES_SRC} tests/read_sql.cpp)
        target_compile_definitions(leaderboard PRIVATE)
        target_link_libraries(leaderboard PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
        target_include_directories(leaderboard PRIVATE include)
    endif()
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_options(unit_tests PRIVATE -O3 -mcpu=apple-m1 -flto)
else()
    target_compile_options(unit_tests PRIVATE -O3 -march=native -m64 -fpermissive -flto)
endif()

target_compile_definitions(unit_tests PRIVATE)
target_link_libraries(unit_tests PRIVATE re2 range-v3 fmt Catch2::Catch2WithMain)
target_include_directories(unit_tests PRIVATE include)
