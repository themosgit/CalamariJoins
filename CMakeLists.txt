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

file(GLOB ALL_SRC
    CONFIGURE_DEPENDS
    "src/*.cpp"
)

set(MANOLATES_SRC ${ALL_SRC})

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    add_compile_options(-O3 -mcpu=apple-m1 -flto)
else()
    add_compile_options(-O0 -march=skylake -m64 -mcrc32 -fpermissive)
endif()

add_link_options(-flto)

add_executable(unit_tests ${MANOLATES_SRC} tests/unit_tests.cpp)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_executable(faster ${MANOLATES_SRC} tests/read_sql.cpp)
    target_compile_definitions(faster PRIVATE)
    target_link_libraries(faster PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
    target_include_directories(faster PRIVATE include)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_options(unit_tests PRIVATE -O3 -mcpu=apple-m1 -flto)
else()
    target_compile_options(unit_tests PRIVATE -O0 -march=skylake -m64 -fpermissive)
endif()

target_compile_definitions(unit_tests PRIVATE)
target_link_libraries(unit_tests PRIVATE re2 range-v3 fmt Catch2::Catch2WithMain)
target_include_directories(unit_tests PRIVATE include)
