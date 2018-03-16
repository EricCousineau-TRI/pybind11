/*
  tests/test_numpy_dtypes.cpp -- User defined NumPy dtypes

  Copyright (c) 2018 Eric Cousineau

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include <cstring>

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/operators.h>
#include <pybind11/numpy_dtype_user.h>
#include <pybind11/embed.h>

namespace py = pybind11;

// Trivial string class.
class CustomStr {
public:
    static constexpr int len = 100;
    CustomStr(const char* s) {
        snprintf(buffer, len, "%s", s);
    }
    template <typename Arg, typename... Args>
    CustomStr(const char* fmt, Arg arg, Args... args) {
        snprintf(buffer, len, fmt, arg, args...);
    }
    std::string str() const {
        return buffer;
    }
    bool operator==(const CustomStr& other) const {
        return str() == other.str();
    }
private:
    char buffer[len];
};

PYBIND11_NUMPY_DTYPE_USER(CustomStr);

class Custom {
public:
    Custom() {
        print_created(this);
    }
    ~Custom() {
        print_destroyed(this);
    }
    Custom(double value) : value_{value} {
        print_created(this, value);
    }
    Custom(const Custom& other) {
        value_ = other.value_;
        print_copy_created(this, other.value_);
    }
    Custom& operator=(const Custom& other) {
        print_copy_assigned(this, other.value_);
        value_ = other.value_;
        return *this;
    }
    operator double() const { return value_; }

    Custom operator+(const Custom& rhs) const {
        py::print("add: ", value_, rhs.value_);
        auto tmp = Custom(value_ + rhs.value_);
        py::print(" = ", tmp.value_);
        return tmp;
    }
    Custom& operator+=(const Custom& rhs) {
        py::print("iadd: ", value_, rhs.value_);
        value_ += rhs.value_;
        py::print(" = ", value_);
        return *this;
    }
    Custom operator+(double rhs) const {
        py::print("add: ", value_, rhs);
        auto tmp = Custom(value_ + rhs);
        py::print(" = ", tmp.value_);
        return tmp;
    }
    Custom& operator+=(double rhs) {
        py::print("iadd: ", value_, rhs);
        value_ += rhs;
        py::print(" = ", value_);
        return *this;
    }
    Custom operator*(const Custom& rhs) const { return value_ * rhs.value_; }
    Custom operator-(const Custom& rhs) const { return value_ - rhs.value_; }

    Custom operator-() const { return -value_; }

    CustomStr operator==(const Custom& rhs) const {
        // Return non-boolean dtype.
        return CustomStr("%g == %g", value_, rhs.value_);
    }
    bool operator<(const Custom& rhs) const {
        // Return boolean value.
        return value_ < rhs.value_;
    }

private:
    double value_{};
};

PYBIND11_NUMPY_DTYPE_USER(Custom);

//TEST_SUBMODULE(numpy_dtype_user, m) {
void numpy_dtype_user(py::module m) {
    ConstructorStats::type_fallback([](py::object cls) -> std::type_index* {
        return const_cast<std::type_index*>(py::detail::dtype_info::find_entry(cls));
    });

    try { py::module::import("numpy"); }
    catch (...) { return; }

    // Bare, minimal type.
    py::dtype_user<CustomStr>(m, "CustomStr")
        .def(py::init<const char*>())
        .def("__str__", &CustomStr::str)
        .def("__repr__", &CustomStr::str)
        .def_ufunc_cast([](const CustomStr& in) -> double {
            py::pybind11_fail("Cannot cast");
        });

    // Not explicitly convertible: `double`
    // Explicitly convertible: `char[4]` (artibrary)
    using char4 = std::array<char, 4>;

    // Somewhat more expressive.
    py::dtype_user<Custom>(m, "Custom")
        .def(py::init())
        // ISSUE: Adding a copy constructor here is actually causing recursion...
        .def(py::init<double>())
        .def("__repr__", [](const Custom* self) {
            return py::str("<Custom({})>").format(double{*self});
        })
        .def("__str__", [](const Custom* self) {
            return py::str("Custom({})").format(double{*self});
        })
        // Test referencing.
        .def("self", [](Custom* self) { return self; }, py::return_value_policy::reference)
        // Casting.
        .def_ufunc_cast([](const double& in) -> Custom { return in; })
        .def_ufunc_cast([](const Custom& in) -> double { return in; })
        .def_ufunc_cast([](const char4& in) -> Custom { return Custom(4); }, true)
        .def_ufunc_cast([](const Custom& in) -> char4 { return {{'a', 'b', 'c', 'd'}}; }, true)
        // TODO(eric.cousineau): Figure out type for implicit coercion.
        // Operators + ufuncs, with some just-operators (e.g. in-place)
        .def_ufunc(py::self + py::self)
        .def(py::self += py::self)
        .def_ufunc(py::self + double{})
        .def(py::self += double{})
        .def_ufunc(py::self * py::self)
        .def_ufunc(py::self - py::self)
        .def_ufunc(-py::self)
        .def_ufunc(py::self == py::self)
        .def_ufunc(py::self < py::self);

    m.def("same", [](const Custom& a, const Custom& b) {
        return double{a} == double{b};
    });
    m.def("same", [](const CustomStr& a, const CustomStr& b) {
        return a == b;
    });
}

void bind_ConstructorStats(py::module &m);

int main() {
    py::scoped_interpreter guard;

    py::module m("pybind11_tests");
    bind_ConstructorStats(m);
    numpy_dtype_user(m.def_submodule("numpy_dtype_user"));
    // py::module s = m.def_submodule("numpy_dtype_user");
    // py::module o = py::module::import("numpy_dtype_user");
    // s.attr("__dict__").attr("update")(o.attr("__dict__"));

    py::str file = "python/pybind11/tests/test_numpy_dtype_user.py";
    py::print(file);
    py::module mm("__main__");
    mm.attr("__file__") = file;
    py::eval_file(file);
    return 0;
}
