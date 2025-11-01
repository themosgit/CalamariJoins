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

FetchContent_Declare(
    duckdb
    URL https://github.com/duckdb/duckdb/archive/refs/tags/v1.2.1.tar.gz
)
set(ENABLE_SANITIZER OFF)
set(ENABLE_UBSAN OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|powerpc|ppc64|ppc64le")
    message("Disabling jemalloc extension of DuckDB on Power.")
    set(SKIP_EXTENSIONS jemalloc)
endif()
FetchContent_MakeAvailable(duckdb)

find_package(LLVM REQUIRED CONFIG)

# Include all sources from /src directory. CONFIGURE_DEPENDS can be unreliable.
# Try re-running cmake in case changes are not recognized.
file(GLOB ALL_SRC
    CONFIGURE_DEPENDS
    "src/*.cpp"
)

set(SIGMODPC_SRC ${ALL_SRC})
list(REMOVE_ITEM SIGMODPC_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/execute_opt.cpp"
)

set(MANOLATES_SRC ${ALL_SRC})
list(REMOVE_ITEM MANOLATES_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/execute.cpp"
)

# Enable server-specific compiler optimizations.
# Use march=native for all but Power servers, which results in the following error:
#    clang++-18: error: unsupported option '-march=' for target 'powerpc64le-unknown-linux-gnu'
# This flag works on other Power systems, but for now, we disable march=native on all Power machines.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    add_compile_options(-O3 -mcpu=apple-m1)
else()
    add_compile_options(-O3 -march=native -m64 -mcrc32 -fpermissive)
endif()

# Targets
add_executable(run ${SIGMODPC_SRC} tests/read_sql.cpp)
add_executable(unit_tests ${SIGMODPC_SRC} tests/unit_tests.cpp)
add_executable(hopscotch_tests ${MANOLATES_SRC} tests/hopscotch_tests.cpp)
add_executable(robinhood_tests ${MANOLATES_SRC} tests/robinhood_tests.cpp)
add_executable(cuckoo_tests ${MANOLATES_SRC} tests/cuckoo_tests.cpp)
add_executable(build_database tests/build_database.cpp)

# For all executables
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(build_database PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(unit_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_executable(build_cache ${SIGMODPC_SRC} tests/read_sql.cpp)
    target_link_libraries(build_cache PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser duckdb)
    target_compile_definitions(build_cache PRIVATE TEAMOPT_USE_DUCKDB TEAMOPT_BUILD_CACHE)
    target_include_directories(build_cache PRIVATE include)

    add_executable(fast ${SIGMODPC_SRC} tests/read_sql.cpp)
    target_compile_definitions(fast PRIVATE HASHTABLE=0)
    target_link_libraries(fast PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
    target_include_directories(fast PRIVATE include)

    add_executable(hopscotch_base ${SIGMODPC_SRC} tests/read_sql.cpp)
    target_compile_definitions(hopscotch_base PRIVATE HASHTABLE=1)
    target_link_libraries(hopscotch_base PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
    target_include_directories(hopscotch_base PRIVATE include)

    add_executable(robinhood_base ${SIGMODPC_SRC} tests/read_sql.cpp)
    target_compile_definitions(robinhood_base PRIVATE HASHTABLE=2)
    target_link_libraries(robinhood_base PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
    target_include_directories(robinhood_base PRIVATE include)

    add_executable(cuckoo_base ${SIGMODPC_SRC} tests/read_sql.cpp)
    target_compile_definitions(cuckoo_base PRIVATE HASHTABLE=3)
    target_link_libraries(cuckoo_base PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
    target_include_directories(cuckoo_base PRIVATE include)

    add_executable(hopscotch ${MANOLATES_SRC} tests/read_sql.cpp)
    target_compile_definitions(hopscotch PRIVATE HASHTABLE=1)
    target_link_libraries(hopscotch PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
    target_include_directories(hopscotch PRIVATE include)

    add_executable(robinhood ${MANOLATES_SRC} tests/read_sql.cpp)
    target_compile_definitions(robinhood PRIVATE HASHTABLE=2)
    target_link_libraries(robinhood PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
    target_include_directories(robinhood PRIVATE include)

    add_executable(cuckoo ${MANOLATES_SRC} tests/read_sql.cpp)
    target_compile_definitions(cuckoo PRIVATE HASHTABLE=3)
    target_link_libraries(cuckoo PRIVATE re2 fmt range-v3 nlohmann_json::nlohmann_json sqlparser)
    target_include_directories(cuckoo PRIVATE include)

endif()

target_link_libraries(build_database PRIVATE fmt duckdb)
target_include_directories(build_database PRIVATE include)

target_link_libraries(unit_tests PRIVATE re2 range-v3 fmt Catch2::Catch2WithMain)
target_include_directories(unit_tests PRIVATE include)

target_compile_definitions(hopscotch_tests PRIVATE HASHTABLE=1)
target_link_libraries(hopscotch_tests PRIVATE re2 range-v3 fmt Catch2::Catch2WithMain)
target_include_directories(hopscotch_tests PRIVATE include)

target_compile_definitions(robinhood_tests PRIVATE HASHTABLE=2)
target_link_libraries(robinhood_tests PRIVATE re2 range-v3 fmt Catch2::Catch2WithMain)
target_include_directories(robinhood_tests PRIVATE include)

target_compile_definitions(cuckoo_tests PRIVATE HASHTABLE=3)
target_link_libraries(cuckoo_tests PRIVATE re2 range-v3 fmt Catch2::Catch2WithMain)
target_include_directories(cuckoo_tests PRIVATE include)


