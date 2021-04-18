/*
    tests/test_ownership_transfer.cpp -- test ownership transfer semantics.

    Copyright (c) 2017 Eric Cousineau <eric.cousineau@tri.global>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#if defined(_MSC_VER) && _MSC_VER < 1910
#  pragma warning(disable: 4702) // unreachable code in system header
#endif

#include <memory>

#include "pybind11_tests.h"
#include "object.h"

enum Label : int {
  BaseBadLabel,
  ChildBadLabel,
  BaseLabel,
  ChildLabel,

  BaseBadUniqueLabel,
  ChildBadUniqueLabel,
  BaseUniqueLabel,
  ChildUniqueLabel,
};

// For attaching instances of `ConstructorStats`.
template <int label>
class Stats {};

struct Base1 {
    Base1(int i) : i(i) { }
    int foo() { return i; }
    int i;
};

struct Base2a {
    Base2a(int i) : i(i) { }
    int bar() { return i; }
    int i;
};

template <typename T, typename Ptr = std::unique_ptr<T>>
class Container {
public:
    Container(Ptr ptr)
        : ptr_(std::move(ptr)) {}
    T* get() const { return ptr_.get(); }
    Ptr release() { return std::move(ptr_); }

    static void def(py::module &m, const std::string& name) {
        py::class_<Container>(m, name.c_str())
            .def(py::init<Ptr>())
            .def("get", &Container::get)
            .def("release", &Container::release);
    }
private:
    Ptr ptr_;
};

template <int label>
class DefineBase {
 public:
  DefineBase(int value)
      : value_(value) {
    track_created(this, value);
  }
  // clang does not like having an implicit copy constructor when the
  // class is virtual (and rightly so).
  DefineBase(const DefineBase&) = delete;
  virtual ~DefineBase() {
    track_destroyed(this);
  }
  virtual int value() const { return value_; }
 private:
  int value_{};
};

template <int label>
class DefineBaseContainer {
 public:
  using T = DefineBase<label>;
  DefineBaseContainer(std::shared_ptr<T> obj)
      : obj_(obj) {}
  std::shared_ptr<T> get() const { return obj_; }
  std::shared_ptr<T> release() { return std::move(obj_); }
 private:
  std::shared_ptr<T> obj_;
};

template <int label>
class DefineBaseUniqueContainer {
 public:
  using T = DefineBase<label>;
  DefineBaseUniqueContainer(std::unique_ptr<T> obj)
      : obj_(std::move(obj)) {}
  T* get() const { return obj_.get(); }
  std::unique_ptr<T> release() { return std::move(obj_); }
 private:
  std::unique_ptr<T> obj_;
};

template <int label>
class DefinePyBase : public DefineBase<label> {
 public:
  using BaseT = DefineBase<label>;
  using BaseT::BaseT;
  int value() const override {
    PYBIND11_OVERLOAD(int, BaseT, value);
  }
};

template <int label>
class DefinePyBaseWrapped : public py::wrapper<DefineBase<label>> {
 public:
  using BaseT = py::wrapper<DefineBase<label>>;
  using BaseT::BaseT;
  int value() const override {
    PYBIND11_OVERLOAD(int, BaseT, value);
  }
};

// BaseBad - No wrapper alias.
using BaseBad = DefineBase<BaseBadLabel>;
using BaseBadContainer = DefineBaseContainer<BaseBadLabel>;
using ChildBadStats = Stats<ChildBadLabel>;

// Base - wrapper alias used in pybind definition.
using Base = DefineBase<BaseLabel>;
using PyBase = DefinePyBase<BaseLabel>;
using BaseContainer = DefineBaseContainer<BaseLabel>;
using ChildStats = Stats<ChildLabel>;

// - Unique Ptr
// BaseBad - No wrapper alias.
using BaseBadUnique = DefineBase<BaseBadUniqueLabel>;
using BaseBadUniqueContainer = DefineBaseUniqueContainer<BaseBadUniqueLabel>;
using ChildBadUniqueStats = Stats<ChildBadUniqueLabel>;

// Base - wrapper alias used directly.
using BaseUnique = DefineBase<BaseUniqueLabel>;
using PyBaseUnique = DefinePyBaseWrapped<BaseUniqueLabel>;
using BaseUniqueContainer = DefineBaseUniqueContainer<BaseUniqueLabel>;
using ChildUniqueStats = Stats<ChildUniqueLabel>;

class PyInstanceStats {
 public:
  PyInstanceStats(ConstructorStats& cstats, py::handle h)
    : cstats_(cstats),
      h_(h) {}
  void track_created() {
    cstats_.created(h_.ptr());
    cstats_.value(py::str(h_).cast<std::string>());
  }
  void track_destroyed() {
    cstats_.destroyed(h_.ptr());
  }
 private:
  ConstructorStats& cstats_;
  py::handle h_;
};

PyInstanceStats get_instance_cstats(ConstructorStats& cstats, py::handle h) {
  return PyInstanceStats(cstats, h);
}

template <typename C, typename... Args>
using class_shared_ = py::class_<C, Args..., std::shared_ptr<C>>;

template <typename... Args>
using class_unique_ = py::class_<Args...>;

class UniquePtrHeld {
public:
    UniquePtrHeld() = delete;
    UniquePtrHeld(const UniquePtrHeld&) = delete;
    UniquePtrHeld(UniquePtrHeld&&) = delete;

    UniquePtrHeld(int value)
        : value_(value) {
        print_created(this, value);
    }
    ~UniquePtrHeld() {
        print_destroyed(this);
    }
    int value() const { return value_; }
private:
    int value_{};
};

TEST_SUBMODULE(drake_smart_ptr, m) {
  // For Drake issue: https://github.com/RobotLocomotion/drake/issues/9398
  m.def("test_pointer_caster", []() -> bool {
      UserType a;
      UserType *a_ptr = &a;
      py::object o = py::cast(&a); // Rvalue
      py::object o1 = py::cast(a_ptr); // Non-rvalue
      return (py::cast<UserType*>(o) == a_ptr && py::cast<UserType*>(o1) == a_ptr);
  });

  Container<Base1>::def(m, "ContainerBase1");

  Container<Base2a, std::shared_ptr<Base2a>>::def(m, "ContainerBase2a");

  // Test #1922 (drake#11424).
  class ExampleVirt2 {
      public:
          virtual ~ExampleVirt2() = default;
          virtual std::string get_name() const { return "ExampleVirt2"; }
      };
  class PyExampleVirt2 : public ExampleVirt2 {
  public:
      std::string get_name() const override {
          PYBIND11_OVERLOAD(std::string, ExampleVirt2, get_name, );
      }
  };
  py::class_<ExampleVirt2, PyExampleVirt2>(m, "ExampleVirt2")
      .def(py::init())
      .def("get_name", &ExampleVirt2::get_name);
  m.def("example_virt2_get_name",
      [](const ExampleVirt2& obj) { return obj.get_name(); });

  // No alias - will not have lifetime extended.
  class_shared_<BaseBad>(m, "BaseBad")
      .def(py::init<int>())
      .def("value", &BaseBad::value);
  class_shared_<BaseBadContainer>(m, "BaseBadContainer")
      .def(py::init<std::shared_ptr<BaseBad>>())
      .def("get", &BaseBadContainer::get)
      .def("release", &BaseBadContainer::release);
  class_shared_<ChildBadStats>(m, "ChildBadStats");

  // Has alias - will have lifetime extended.
  class_shared_<Base, py::wrapper<PyBase>>(m, "Base")
      .def(py::init<int>())
      // Factory method for alias.
      .def(py::init([]() { return new py::wrapper<PyBase>(10); }))
      .def("value", &Base::value);
  class_shared_<BaseContainer>(m, "BaseContainer")
      .def(py::init<std::shared_ptr<Base>>())
      .def("get", &BaseContainer::get)
      .def("release", &BaseContainer::release);
  class_shared_<ChildStats>(m, "ChildStats");

  class_unique_<BaseBadUnique>(m, "BaseBadUnique")
      .def(py::init<int>())
      .def("value", &BaseBadUnique::value);
  class_unique_<BaseBadUniqueContainer>(m, "BaseBadUniqueContainer")
      .def(py::init<std::unique_ptr<BaseBadUnique>>())
      .def("get", &BaseBadUniqueContainer::get)
      .def("release", &BaseBadUniqueContainer::release);
  class_unique_<ChildBadUniqueStats>(m, "ChildBadUniqueStats");

  class_unique_<BaseUnique, PyBaseUnique>(m, "BaseUnique")
      .def(py::init<int>())
      // Factory method.
      .def(py::init([]() { return new PyBaseUnique(10); }))
      .def("value", &BaseUnique::value);
  class_unique_<BaseUniqueContainer>(m, "BaseUniqueContainer")
      .def(py::init<std::unique_ptr<BaseUnique>>())
      .def("get", &BaseUniqueContainer::get)
      .def("release", &BaseUniqueContainer::release);
  class_unique_<ChildUniqueStats>(m, "ChildUniqueStats");

  class_shared_<PyInstanceStats>(m, "InstanceStats")
      .def(py::init<ConstructorStats&, py::handle>())
      .def("track_created", &PyInstanceStats::track_created)
      .def("track_destroyed", &PyInstanceStats::track_destroyed);
  m.def("get_instance_cstats", &get_instance_cstats);


    py::class_<UniquePtrHeld>(m, "UniquePtrHeld")
        .def(py::init<int>())
        .def("value", &UniquePtrHeld::value);

    class UniquePtrOther {};
    py::class_<UniquePtrOther>(m, "UniquePtrOther")
        .def(py::init<>());

    m.def("unique_ptr_pass_through",
        [](std::unique_ptr<UniquePtrHeld> obj) {
            return obj;
        });
    m.def("unique_ptr_terminal",
        [](std::unique_ptr<UniquePtrHeld> obj) {
            obj.reset();
            return nullptr;
        });

    // Check traits in a concise manner.
    static_assert(
        py::detail::move_common<std::unique_ptr<UniquePtrHeld>>::value,
        "This must be true.");

    // Guarantee API works as expected.
    m.def("unique_ptr_pass_through_cast_from_py",
        [](py::object obj_py) {
            auto obj =
                py::cast<std::unique_ptr<UniquePtrHeld>>(std::move(obj_py));
            return obj;
        });
    m.def("unique_ptr_pass_through_move_from_py",
        [](py::object obj_py) {
            return py::move<std::unique_ptr<UniquePtrHeld>>(std::move(obj_py));
        });

    m.def("unique_ptr_pass_through_move_to_py",
        [](std::unique_ptr<UniquePtrHeld> obj) {
            return py::move(std::move(obj));
        });

    m.def("unique_ptr_pass_through_cast_to_py",
        [](std::unique_ptr<UniquePtrHeld> obj) {
            return py::cast(std::move(obj));
        });

    class FirstT {};
    py::class_<FirstT>(m, "FirstT")
        .def(py::init());
    class SecondT {};
    py::class_<SecondT>(m, "SecondT")
        .def(py::init());

    m.def("unique_ptr_overload",
        [](std::unique_ptr<UniquePtrHeld> obj, FirstT) {
            py::dict out;
            out["obj"] = py::cast(std::move(obj));
            out["overload"] = 1;
            return out;
        });
    m.def("unique_ptr_overload",
        [](std::unique_ptr<UniquePtrHeld> obj, SecondT) {
            py::dict out;
            out["obj"] = py::cast(std::move(obj));
            out["overload"] = 2;
            return out;
        });

    // Ensure class is non-empty, so it's easier to detect double-free
    // corruption. (If empty, this may be harder to see easily.)
    struct SharedPtrHeld { int value = 10; };
    py::class_<SharedPtrHeld, std::shared_ptr<SharedPtrHeld>>(m, "SharedPtrHeld")
        .def(py::init<>());
    m.def("shared_ptr_held_in_unique_ptr",
        []() {
            return std::unique_ptr<SharedPtrHeld>(new SharedPtrHeld());
        });
    m.def("shared_ptr_held_func",
        [](std::shared_ptr<SharedPtrHeld> obj) {
            return obj != nullptr && obj->value == 10;
        });

    // Test passing ownership of registered, but unowned, C++ instances back to
    // Python. This happens when a raw pointer is passed first, and then
    // ownership is transfered.
    struct UniquePtrHeldContainer {
        UniquePtrHeldContainer() {
            value_.reset(new UniquePtrHeld(10));
        }
        UniquePtrHeld* get() const {
            return value_.get();
        }
        using Ptr = std::unique_ptr<UniquePtrHeld>;
        Ptr reset(Ptr to) {
            Ptr from = std::move(value_);
            value_ = std::move(to);
            return from;
        }
        std::unique_ptr<UniquePtrHeld> value_;
    };
    py::class_<UniquePtrHeldContainer>(m, "UniquePtrHeldContainer")
        .def(py::init())
        .def("get", &UniquePtrHeldContainer::get, py::return_value_policy::reference_internal)
        .def("reset", &UniquePtrHeldContainer::reset);
}
