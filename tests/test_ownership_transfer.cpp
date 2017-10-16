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

//#include "python/pybind11_include/pybind11/pybind11.h"

enum Label : int {
  BaseBadLabel = 0,
  ChildBadLabel = 1,
  BaseLabel = 2,
  ChildLabel = 3,
};

// For attaching instances of `ConstructorStats`.
template <int label>
class Stats {};


template <int label>
class DefineBase {
 public:
  DefineBase(int value)
      : value_(value) {
    track_created(this, value);
  }
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
 private:
  std::shared_ptr<T> obj_;
};

// BaseBad - No wrapper alias.
typedef DefineBase<BaseBadLabel> BaseBad;
typedef DefineBaseContainer<BaseBadLabel> BaseBadContainer;
typedef Stats<ChildBadLabel> ChildBadStats;

// Base - with wrapper alias.
typedef DefineBase<BaseLabel> Base;
typedef DefineBaseContainer<BaseLabel> BaseContainer;
typedef Stats<ChildLabel> ChildStats;

class PyBase : public py::wrapper<Base> {
 public:
  using BaseT = py::wrapper<Base>;
  using BaseT::BaseT;
  int value() const override {
    PYBIND11_OVERLOAD(int, Base, value);
  }
};

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

TEST_SUBMODULE(ownership_transfer, m) {
  class_shared_<BaseBad>(m, "BaseBad")
      .def(py::init<int>())
      .def("value", &BaseBad::value);
  class_shared_<BaseBadContainer>(m, "BaseBadContainer")
      .def(py::init<std::shared_ptr<BaseBad>>())
      .def("get", &BaseBadContainer::get);
  class_shared_<ChildBadStats>(m, "ChildBadStats");

  class_shared_<Base, PyBase>(m, "Base")
      .def(py::init<int>())
      .def("value", &Base::value);
  class_shared_<BaseContainer>(m, "BaseContainer")
      .def(py::init<std::shared_ptr<Base>>())
      .def("get", &BaseContainer::get);
  class_shared_<ChildStats>(m, "ChildStats");

  class_shared_<PyInstanceStats>(m, "InstanceStats")
      .def(py::init<ConstructorStats&, py::handle>())
      .def("track_created", &PyInstanceStats::track_created)
      .def("track_destroyed", &PyInstanceStats::track_destroyed);
  m.def("get_instance_cstats", &get_instance_cstats);
}
