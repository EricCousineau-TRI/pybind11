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
private:
    char buffer[len];
};

PYBIND11_NUMPY_DTYPE_USER(CustomStr);

class Custom {
public:
    Custom() {
        track_created(this);
    }
    ~Custom() {
        track_destroyed(this);
    }
    Custom(double value) : value_{value} {
        track_created(this, value);
    }
    Custom(const Custom& other) {
      value_ = other.value_;
      track_copy_created(this);
    }
    Custom& operator=(const Custom& other) {
        track_copy_assigned(this);
        value_ = other.value_;
        return *this;
    }
    operator double() const { return value_; }

    Custom operator+(const Custom& rhs) const { return Custom(*this) += rhs.value_; }
    Custom& operator+=(const Custom& rhs) { value_ += rhs.value_; return *this; }
    Custom operator*(const Custom& rhs) const { return value_ * rhs.value_; }
    Custom operator-(const Custom& rhs) const { return value_ - rhs.value_; }

    Custom operator-() const { return -value_; }

    CustomStr operator==(const Custom& rhs) const {
        // Return non-boolean dtype.
        return CustomStr("%g == %g", value_, rhs.value_);
    }
    Custom operator<(const Custom& rhs) const {
        // Return boolean value.
        return value_ < rhs.value_;
    }

private:
    double value_{};
};

PYBIND11_NUMPY_DTYPE_USER(Custom);

TEST_SUBMODULE(numpy_dtype_user, m) {
    try { py::module::import("numpy"); }
    catch (...) { return; }

    // Bare, minimal type.
    py::dtype_user<CustomStr>(m, "CustomStr")
        .def(py::init<const char*>())
        .def("__str__", &CustomStr::str)
        .def("__repr__", &CustomStr::str);

    // Somewhat more expressive.
    py::dtype_user<Custom>(m, "Custom")
        .def("__repr__", [](const Custom* self) {
            return py::str("<Custom({})>").format(double{*self});
        })
        .def("__str__", [](const Custom* self) {
            return py::str("Custom({})").format(double{*self});
        })
        // Test referencing.
        .def("self", [](Custom* self) { return self; }, py::return_value_policy::reference)
        // Operators + ufuncs, with some just-operators (e.g. in-place)
        .def_ufunc_cast([](const Custom& in) -> double { return in; })
        .def_ufunc_cast([](const double& in) -> Custom { return in; })
        .def_ufunc(py::self + py::self)
        .def(py::self += py::self)
        .def_ufunc(py::self * py::self)
        .def_ufunc(py::self - py::self)
        .def_ufunc(-py::self)
        .def_ufunc(py::self == py::self)
        .def_ufunc(py::self < py::self);
}
