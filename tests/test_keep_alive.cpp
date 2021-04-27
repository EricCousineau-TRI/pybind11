#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/stl.h>

namespace {

class ExampleOther {};

class ExampleReturn {};

static ExampleReturn return_;

ExampleReturn* example_free_func(ExampleOther*) { return &return_; }

class ExampleSelf {
public:
    ExampleSelf() {}
    ExampleSelf(ExampleOther*) {}

    static ExampleSelf make_str(ExampleOther*, const std::string&) {
        return ExampleSelf{};
    }
    static ExampleSelf make_int(ExampleOther*, int) {
        return ExampleSelf{};
    }

    ExampleReturn* example_method(const ExampleOther&) { return &return_; }
    static ExampleReturn* example_static_method(const ExampleOther&) { return &return_; }
};

}   // namespace

TEST_SUBMODULE(keep_alive, m) {
    using rvp = py::return_value_policy;
    m.def(
        "example_free_func",
        &example_free_func,
        rvp::reference,
        py::keep_alive<0, 1>());

    py::class_<ExampleOther>(m, "ExampleOther")
        .def("__repr__", [](const ExampleOther*) { return "<ExampleOther>"; })
        .def(py::init());
    py::class_<ExampleReturn>(m, "ExampleReturn")
        .def("__repr__", [](const ExampleReturn*) { return "<ExampleReturn>"; });

    py::class_<ExampleSelf>(m, "ExampleSelf")
        .def("__repr__", [](const ExampleSelf*) { return "<ExampleSelf>"; })
        // Constructor.
        .def(
            py::init<ExampleOther*>(),
            py::keep_alive<1, 2>())
        // Factory.
        .def(
            py::init(&ExampleSelf::make_str),
            py::keep_alive<1, 2>())
        // Factory, use `return` value?
        .def(
            py::init(&ExampleSelf::make_int),
            py::keep_alive<0, 2>())
        // Method.
        .def(
            "example_method",
            &ExampleSelf::example_method,
            py::keep_alive<1, 2>())
        // Static.
        .def_static(
            "example_static_method",
            &ExampleSelf::example_static_method,
            py::keep_alive<0, 1>());
}
