/*
  tests/test_numpy_dtypes.cpp -- User defined NumPy dtypes

  Copyright (c) 2018 Eric Cousineau

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

#include <cstring>

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy_dtype_user.h>
#include <pybind11/embed.h>

using std::make_unique;
using std::string;
using std::unique_ptr;

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

// Basic structure, meant to be an implicitly convertible value for `Custom`.
// Would have used a struct type, but the scalars are only tuples.
struct SimpleStruct {
    double value;

    SimpleStruct(double value_in) : value(value_in) {}
};

PYBIND11_NUMPY_DTYPE_USER(SimpleStruct);

template <typename T>
void clone(const unique_ptr<T>& src, unique_ptr<T>& dst) {
    if (!src)
        dst.reset();
    else
        dst = make_unique<T>(*src);
}

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
    Custom(double value, string str)
        : value_{value}, str_{make_unique<string>(str)} {
        print_created(this, value, str);
    }
    Custom(const Custom& other) {
        value_ = other.value_;
        clone(other.str_, str_);
        print_copy_created(this, other.value_);
    }
    Custom(const SimpleStruct& other) {
        value_ = other.value;
        print_copy_created(this, other);
    }
    Custom& operator=(const Custom& other) {
        print_copy_assigned(this, other.value_);
        value_ = other.value_;
        clone(other.str_, str_);
        return *this;
    }
    operator double() const { return value_; }
    operator SimpleStruct() const { return {value_}; }

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
        return CustomStr("%g == %g && %s == %s", value_, rhs.value_, str().c_str(), rhs.str().c_str());
    }
    bool operator<(const Custom& rhs) const {
        // Return boolean value.
        return value_ < rhs.value_;
    }

    std::string str() const {
        if (str_)
            return *str_;
        else
            return {};
    }

private:
    double value_{};
    // Use non-trivial data object, but something that is memcpy-movable.
    std::unique_ptr<string> str_;
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
    auto ss_str = [](const SimpleStruct& self) {
        return py::str("SimpleStruct({})").format(self.value);
    };
    py::dtype_user<SimpleStruct>(m, "SimpleStruct")
        .def(py::init<double>())
        .def("__str__", ss_str)
        .def("__repr__", ss_str);

    // Somewhat more expressive.
    py::dtype_user<Custom>(m, "Custom")
        .def(py::init())
        .def(py::init<double>())
        .def(py::init<const SimpleStruct&>())
        .def(py::init<Custom>())
        .def("__repr__", [](const Custom* self) {
            return py::str("<Custom({})>").format(double{*self});
        })
        .def("__str__", [](const Custom* self) {
            return py::str("Custom({})").format(double{*self});
        })
        // Test referencing.
        .def("self", [](Custom* self) { return self; }, py::return_value_policy::reference)
        // Casting.
        // - Explicit casting (e.g., we have additional arguments).
        .def_ufunc_cast([](const double& in) -> Custom { return in; })
        .def_ufunc_cast(&Custom::operator double)
        // - Implicit coercion + conversion
        .def_ufunc_cast([](const SimpleStruct& in) -> Custom { return in; }, true)
        .def_ufunc_cast(&Custom::operator SimpleStruct, true)
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
