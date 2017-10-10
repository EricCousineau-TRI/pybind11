cc_binary(
    name = "test_unique_ptr_cast",
    srcs = [
        "test_unique_ptr_cast.cc",
    ],
    deps = [
        "//python:pybind11_include",
    ],
)

cc_binary(
    name = "test_shared_ptr_owner",
    srcs = [
        "test_shared_ptr_owner.cc",
    ],
    deps = [
        "//python:pybind11_include",
    ],
)

cc_binary(
    name = "test_ref_count",
    srcs = [
        "test_ref_count.cc",
    ],
    deps = [
        "//python:pybind11_include",
    ],
)
cc_library(
    name = "base",
    srcs = [
        "constructor_stats.h",
        "pybind11_tests.h",
        "pybind11_tests.cpp",
    ],
    deps = [
        "//python:pybind11_include",
    ],
)

cc_binary(
    name = "test_copy_move",
    srcs = [
        "test_copy_move.cpp",
    ],
    deps = [
        ":base",
    ],
)
