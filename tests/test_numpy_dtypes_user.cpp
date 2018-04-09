/*
  tests/test_numpy_dtypes.cpp -- User defined NumPy dtypes

  Copyright (c) 2018 Eric Cousineau

  All rights reserved. Use of this source code is governed by a
  BSD-style license that can be found in the LICENSE file.
*/

// TODO(eric.cousineau): See what mixing non-dtypes look like for pybind...

#include <cstring>

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy_dtypes_user.h>

using std::string;
using std::unique_ptr;

namespace py = pybind11;

namespace {

// Trivial string class.
class CustomStr {
public:
    static constexpr int len = 100;
    CustomStr(const char* s) {
        snprintf(buffer.data(), len, "%s", s);
        dummy.reset(new int(1000));
    }
    template <typename Arg, typename... Args>
    CustomStr(const char* fmt, Arg arg, Args... args) {
        snprintf(buffer.data(), len, fmt, arg, args...);
        dummy.reset(new int(1000));
    }
    CustomStr(const CustomStr&) = default;
    CustomStr& operator=(const CustomStr&) = default;
    CustomStr(CustomStr&&) = default;
    CustomStr& operator=(CustomStr&&) = default;
    std::string str() const {
        return buffer.data();
    }
    bool operator==(const CustomStr& other) const {
        return str() == other.str();
    }
private:
    std::array<char, len> buffer;
    // Data member to ensure that we do not get segfaults when carrying around `shared_ptr`s.
    std::shared_ptr<int> dummy;
};

// Basic structure, meant to be an implicitly convertible value for `Custom`.
// Would have used a struct type, but the scalars are only tuples.
struct SimpleStruct {
    double value;

    SimpleStruct(double value_in) : value(value_in) {}
};

// Define operations to return fixed values.
double operator+(double, SimpleStruct) {
    return -1;
}
double operator+(SimpleStruct, SimpleStruct) {
    return 0;
}
double operator+(SimpleStruct, double) {
    return 1;
}

// Clones a unique_ptr using type's copy constructor.
template <typename T>
void clone(const unique_ptr<T>& src, unique_ptr<T>& dst) {
    if (!src)
        dst.reset();
    else
        dst.reset(new T(*src));
}

class Custom {
public:
    Custom() {
        print_created(this);
    }
    ~Custom() {
        track_destroyed(this);
        // TODO(eric.cousineau): Track down why this causes a segfault.
        //print_destroyed(this);
    }
    Custom(double value) : value_{value} {
        print_created(this, value);
    }
    Custom(double value, string str)
        : value_{value}, str_{unique_ptr<string>(new string(str))} {
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

    double value() const { return value_; }

    operator SimpleStruct() const { return {value_}; }

    Custom operator+(const Custom& rhs) const {
        auto tmp = Custom(value_ + rhs.value_);
        return tmp;
    }
    Custom& operator+=(const Custom& rhs) {
        value_ += rhs.value_;
        return *this;
    }
    Custom operator+(double rhs) const {
        auto tmp = Custom(value_ + rhs);
        return tmp;
    }
    Custom& operator+=(double rhs) {
        value_ += rhs;
        return *this;
    }
    Custom operator*(const Custom& rhs) const { return value_ * rhs.value_; }
    Custom operator-(const Custom& rhs) const { return value_ - rhs.value_; }

    Custom operator-() const { return -value_; }

    Custom operator/(const Custom& rhs) const {
        return Custom(value_ / rhs.value_);
    }

    bool same_as(const Custom& rhs) const {
        return value_ == rhs.value_ && str() == rhs.str();
    }
    CustomStr operator==(const Custom& rhs) const {
        // Return non-boolean dtype.
        return CustomStr("%g == %g && '%s' == '%s'", value_, rhs.value_, str().c_str(), rhs.str().c_str());
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

Custom operator+(double a, const Custom& b) {
    return Custom(a) += b;
}

// Define `dtype=object` types, and ensure the operations correctly reflect these.
struct ObjectA {};
struct ObjectB {};

Custom operator+(const Custom& a, const ObjectA& b) {
    return Custom(1000);
}
Custom operator+(const Custom& a, const ObjectB& b) {
    return Custom(9999);
}

}  // namespace

#if defined(PYBIND11_CPP14)

PYBIND11_NUMPY_DTYPE_USER(CustomStr);
PYBIND11_NUMPY_DTYPE_USER(SimpleStruct);
PYBIND11_NUMPY_DTYPE_USER(Custom);

PYBIND11_NUMPY_OBJECT_DTYPE(ObjectA);
PYBIND11_NUMPY_OBJECT_DTYPE(ObjectB);

TEST_SUBMODULE(numpy_dtype_user, m) {
    ConstructorStats::type_fallback([](py::object cls) -> std::type_index* {
        return const_cast<std::type_index*>(py::detail::dtype_info::find_entry(cls));
    });

    try { py::module::import("numpy"); }
    catch (...) { return; }

    py::class_<ObjectA>(m, "ObjectA")
        .def(py::init());
    py::class_<ObjectB>(m, "ObjectB")
        .def(py::init());

    // Bare, minimal type.
    py::dtype_user<CustomStr>(m, "CustomStr")
        .def(py::init<const char*>())
        .def("__str__", &CustomStr::str)
        .def("__repr__", &CustomStr::str)
        .def_loop_cast([](const CustomStr&) -> double {
            py::pybind11_fail("Cannot cast");
        })
        .def(py::self == py::self);

    // Not explicitly convertible: `double`
    auto ss_str = [](const SimpleStruct& self) {
        return py::str("SimpleStruct({})").format(self.value);
    };
    py::dtype_user<SimpleStruct> simple_struct(m, "SimpleStruct");
    py::dtype_user<Custom> custom(m, "Custom");
    simple_struct
        .def(py::init<double>())
        .def("__str__", ss_str)
        .def("__repr__", ss_str)
        // Test operator ordering.
        .def_loop(py::self + py::self)
        .def_loop(double{} + py::self)
        .def_loop(py::self + double{})
        .def_loop_cast([](const SimpleStruct& in) -> Custom { return in; }, true);

    custom
        .def(py::init())
        .def(py::init<double>())
        .def(py::init<double, string>())
        .def(py::init<const SimpleStruct&>())
        .def(py::init<Custom>())
        .def("__repr__", [](const Custom* self) {
            return py::str("Custom({}, '{}')").format(self->value(), self->str());
        })
        .def("__str__", [](const Custom* self) {
            return py::str("C<{}, '{}'>").format(self->value(), self->str());
        })
        .def("value", &Custom::value)
        // Test referencing.
        .def("self", [](Custom* self) { return self; }, py::return_value_policy::reference)
        // Casting.
        // N.B. For `np.ones`, we could register a converter from `int64_t` to `Custom`, but this would cause a segfault,
        // because `np.ones` uses `np.copyto(..., casting="unsafe")`, which does *not* respect NPY_NEEDS_INITIALIZATION.
        // - Explicit casting (e.g., we have additional arguments).
        .def_loop_cast([](const Custom& in) { return in.value(); })
        .def_loop_cast([](const double& in) -> Custom { return in; })
            // - Implicit coercion + conversion
        .def_loop_cast(&Custom::operator SimpleStruct, true)
           // - - N.B. This shouldn't be a normal operation (upcasting?), as it may result in data loss.
        .def_loop_cast([](const SimpleStruct& in) -> Custom { return in; }, true)
            // TODO(eric.cousineau): Figure out type for implicit coercion.
            // Operators + ufuncs, with some just-operators (e.g. in-place)
        .def_loop(py::self + py::self)
        .def(py::self += py::self)
        .def_loop(py::self + double{})
        .def_loop(double{} + py::self)
        .def_loop(py::self / py::self)
        .def(py::self += double{})
        .def_loop(py::self * py::self)
        .def_loop(py::self - py::self)
        .def_loop(-py::self)
        .def_loop(py::self == py::self)
        .def_loop(py::self < py::self)
        .def(py::self + ObjectA{})
        .def(py::self + ObjectB{})
        .def_loop(py::dtype_method::dot())
        .def_loop("__pow__", [](const Custom& a, const Custom& b) {
            return CustomStr("%g ^ %g", a.value(), b.value());
        })
        // TOOD(eric.cousineau): Handle pointers too.
        .def_loop("cos", [](const Custom& self) {
            return Custom(cos(self.value()));
        })
        .def_loop("logical_and", [](const Custom&, const Custom&) -> double {
            return 10;
        });
    // Somewhat more expressive.

    auto binary_op = [](const Custom& a, const Custom& b) {
        return CustomStr("%g == %g", a.value(), b.value());
    };
    m.def("binary_op", binary_op);
    py::ufunc(m, "binary_op_loop")
        .def_loop<Custom>(binary_op);

    // N.B. We should not define a boolean operator for `equal`, as NumPy will
    // use this, even if we define it "afterwards", due to how it is stored.

    // `py::vectorize` does not seem to allow custom types due to sfinae constraints :(
    py::ufunc x(m, "same");
    x
        .def_loop<Custom>([](const Custom& a, const Custom& b) {
            return a.same_as(b);
        })
        .def_loop<CustomStr>([](const CustomStr& a, const CustomStr& b) {
            return a == b;
        })
        // Define this for checking other stuff.
        .def_loop<bool>([](bool a, bool b) { return a == b; })
        .def_loop<double>([](double a, double b) { return a == b; })
        .def_loop<py::object>([m](py::object a, py::object b) {
            // Will recurse if we don't have an overload for the given type.
            return m.attr("same")(a, b).cast<bool>();
        });
}

#else  // defined(PYBIND11_CPP14)

TEST_SUBMODULE(numpy_dtype_user, m) {
    m.attr("DISABLED") = true;
}

#endif  // defined(PYBIND11_CPP14)
