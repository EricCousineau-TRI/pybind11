// Purpose: Base what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace std;

class SimpleType {
 public:
    SimpleType(int value)
        : value_(value) {
      cout << "SimpleType::SimpleType()" << endl;
    }
    ~SimpleType() {
      cout << "SimpleType::~SimpleType()" << endl;
    }
    int value() const { return value_; }
 private:
    int value_{};
};

class Base {
 public:
  Base(int value)
      : value_(value) {
    cout << "Base::Base(int)\n";
  }
  virtual ~Base() {
    cout << "Base::~Base()\n";
  }
  virtual int value() const {
    cout << "Base::value()\n";
    return value_;
  }
 private:
  int value_{};
};

class Child : public Base {
 public:
  Child(int value)
     : Base(value) {}
  ~Child() {
    cout << "Child::~Child()\n";
  }
  int value() const override {
    cout << "Child::value()\n";
    return 10 * Base::value();
  }
};

class ChildB : public Base {
 public:
  ChildB(int value)
     : Base(value) {}
  ~ChildB() {
    cout << "ChildB::~ChildB()\n";
  }
  int value() const override {
    cout << "ChildB::value()\n";
    return 10 * Base::value();
  }
};

// TODO(eric.cousineau): Add converter for `is_base<T, trampoline<T>>`, only for
// `cast` (C++ to Python) to handle swapping lifetime control.

// Trampoline class.
class PyBase : public py::trampoline<Base> {
 public:
  typedef py::trampoline<Base> Base;
  using Base::Base;
  ~PyBase() {
    cout << "PyBase::~PyBase()" << endl;
  }
  int value() const override {
    PYBIND11_OVERLOAD(int, Base, value);
  }
};
class PyChild : public py::trampoline<Child> {
 public:
  typedef py::trampoline<Child> Base;
  using Base::Base;
  ~PyChild() {
    cout << "PyChild::~PyChild()" << endl;
  }
  int value() const override {
    PYBIND11_OVERLOAD(int, Child, value);
  }
};
class PyChildB : public py::trampoline<ChildB> {
 public:
  typedef py::trampoline<ChildB> Base;
  using Base::Base;
  ~PyChildB() {
    cout << "PyChildB::~PyChildB()" << endl;
  }
  int value() const override {
    PYBIND11_OVERLOAD(int, ChildB, value);
  }
};

unique_ptr<Base> check_creation(py::function create_obj) {
  // Test getting a pointer.
//  Base* in_test = py::cast<Base*>(obj);
  // Base a terminal pointer.
  // NOTE: This yields a different destructor order.
  // However, the trampoline class destructors should NOT interfere with nominal
  // Python destruction.
  cout << "---\n";
  unique_ptr<Base> fin = py::cast<unique_ptr<Base>>(create_obj());
  fin.reset();
  cout << "---\n";
  // Test pass-through.
  py::object obj = create_obj();
  unique_ptr<Base> in = py::cast<unique_ptr<Base>>(std::move(obj));
  return in;
}

unique_ptr<SimpleType> check_creation_simple(py::function create_obj) {
  cout << "---\n";
  unique_ptr<SimpleType> fin = py::cast<unique_ptr<SimpleType>>(create_obj());
  fin.reset();
  cout << "---\n";
  py::object obj = create_obj();
  unique_ptr<SimpleType> in = py::cast<unique_ptr<SimpleType>>(std::move(obj));
  return in;
}

// TODO(eric.cousineau): If a user uses `object` as a pass in, it should keep the reference count low
// (so that we can steal it, if need be).
// Presently, `pybind11` increases that reference count if `object` is an argument.

// Check casting.
unique_ptr<Base> check_cast_pass_thru(unique_ptr<Base> in) { //py::handle h) { //
//  py::object py_in = py::reinterpret_steal<py::object>(h);
//  auto in = py::cast<unique_ptr<Base>>(std::move(py_in));
  cout << "Pass through: " << in->value()<< endl;
  return in;
}

unique_ptr<Base> check_clone(unique_ptr<Base> in) {
//  auto in = py::cast<unique_ptr<Base>>(std::move(py_in));
  cout << "Clone: " << in->value()<< endl;
  unique_ptr<Base> out(new Base(20 * in->value()));
  return out;
}

unique_ptr<Base> check_new() {
    return make_unique<Base>(10);
}

PYBIND11_MODULE(_move, m) {
  py::class_<Base, PyBase>(m, "Base")
    .def(py::init<int>())
    .def("value", &Base::value);

  py::class_<Child, PyChild, Base>(m, "Child")
      .def(py::init<int>())
      .def("value", &Child::value);

  // NOTE: Not explicit calling `Base` as a base. Relying on Python downcasting via `py_type`.
  py::class_<ChildB, PyChildB>(m, "ChildB")
      .def(py::init<int>())
      .def("value", &ChildB::value);

  m.def("check_creation", &check_creation);
  m.def("check_cast_pass_thru", &check_cast_pass_thru);
  m.def("check_clone", &check_clone);
  m.def("check_new", &check_new);

  // Make sure this setup doesn't botch the usage of `shared_ptr`, compile or run-time.
  class SharedClass {};
  py::class_<SharedClass, shared_ptr<SharedClass>>(m, "SharedClass");

  // Make sure this also still works with non-virtual, non-trampoline types.
  py::class_<SimpleType>(m, "SimpleType")
      .def(py::init<int>())
      .def("value", &SimpleType::value);
  m.def("check_creation_simple", &check_creation_simple);

  auto mdict = m.attr("__dict__");
  py::exec(R"(
class PyExtBase(Base):
    def __init__(self, value):
        Base.__init__(self, value)
        print("PyExtBase.PyExtBase")
    def __del__(self):
        print("PyExtBase.__del__")
    def value(self):
        print("PyExtBase.value")
        return Base.value(self)

class PyExtChild(Child):
    def __init__(self, value):
        Child.__init__(self, value)
        print("PyExtChild.PyExtChild")
    def __del__(self):
        print("PyExtChild.__del__")
    def value(self):
        print("PyExtChild.value")
        return Child.value(self)

class PyExtChildB(ChildB):
    def __init__(self, value):
        ChildB.__init__(self, value)
        print("PyExtChildB.PyExtChildB")
    def __del__(self):
        print("PyExtChildB.__del__")
    def value(self):
        print("PyExtChildB.value")
        return ChildB.value(self)
)", mdict, mdict);

    // Define move container thing
    py::exec(R"(
class PyMove:
    """ Provide a wrapper to permit passing an object to be owned by C++ """
    _is_move_container = True

    def __init__(self, obj):
        assert obj is not None
        self._obj = obj

    def release(self):
        from sys import getrefcount
        obj = self._obj
        self._obj = None
        ref_count = getrefcount(obj)
        # Cannot use `assert ...`, because it will leave a latent reference?
        # Consider a `with` reference?
        if ref_count > 2:
            obj = None
            raise AssertionError("Object reference is not unique, got {} extra references".format(ref_count - 2))
        else:
            assert ref_count == 2
            return obj
)", py::globals(), mdict);
}

// Export this to get access as we desire.
void custom_init_move(py::module& m) {
  PYBIND11_CONCAT(pybind11_init_, _move)(m);
}

void check_pure_cpp_simple() {
  cout << "\n[ check_pure_cpp_simple ]\n";
  py::exec(R"(
def create_obj():
    return [move.SimpleType(256)]
obj = move.check_creation_simple(create_obj)
print(obj.value())
del obj  # Calling `del` since scoping isn't as tight here???
)");
}

void check_pure_cpp() {
  cout << "\n[ check_pure_cpp ]\n";
  py::exec(R"(
def create_obj():
    return [move.Base(10)]
obj = move.check_creation(create_obj)
print(obj.value())
del obj
)");
}

void check_pass_thru() {
    cout << "\n[ check_pure_cpp ]\n";

    py::exec(R"(
obj = move.check_cast_pass_thru([move.Base(20)])
print(obj.value())
del obj

obj = move.check_clone([move.Base(30)])
print(obj.value())
del obj
)");

    if (false) {

    auto m = py::globals()["move"];
    auto base_py_type = m.attr("Base");
    py::object func = m.attr("check_cast_pass_thru");
    py::object obj = base_py_type(10);
    py::handle h = obj.release();
    cout << "ref_count: " << h.ref_count() << endl;
    py::object pass = func(h);  // Does NOT work. Too many references, due to argument packing?

    // ISSUE: For some reason, when packing the argument list, the unique reference lives just within
    // the `simple_collector`, forwarded as a `py::tuple`.
    // When the function call returns, then that object goes out of scope, causing destruction.
//    py::object pass = func(make_unique<Base>(10));  // Rely on casting.

    int value = pass.attr("value")().cast<int>();
    cout << "Value: " << value << endl;

    }

//    py::dict locals;
//    py::exec(R"(
//obj = move.check_cast_pass_thru(move.Base(10))
//# print(move.check_clone(move.Base(20)).value())
//)", py::globals(), locals);
//    // MEMORY LEAK
//    py::exec(R"(
//print("Locals: {}".format(locals()))
//print("Globals: {}".format(globals()))
//)", py::globals(), locals);
//    py::exec(R"(
//print("instance: {}".format(obj))
//print("Obj: {}".format(obj.value()))
//)", py::globals(), locals);
}

void check_py_child() {
  // Check ownership for a Python-extended C++ class.
  cout << "\n[ check_py_child ]\n";
  py::exec(R"(
def create_obj():
    return [move.PyExtBase(20)]
obj = move.check_creation(create_obj)
print(obj.value())
del obj
)");
}

void check_casting() {
  // Check a class which, in C++, derives from the direct type, but not the alias.
  cout << "\n[ check_casting ]\n";
  py::exec(R"(
def create_obj():
    return [move.PyExtChild(30)]
obj = move.check_creation(create_obj)
print(obj.value())
del obj
)");
}

void check_casting_without_explicit_base() {
  // Check a class which, in C++, derives from the direct type, but not the alias.
  cout << "\n[ check_casting_without_explicit_base ]\n";
  py::exec(R"(
def create_obj():
    return [move.PyExtChildB(30)]
obj = move.check_creation(create_obj)
print(obj.value())
del obj
)");
}

int main() {
  {
    py::scoped_interpreter guard{};

    py::module m("_move");
    custom_init_move(m);
    py::globals()["move"] = m;

    check_pass_thru();
    check_pure_cpp_simple();
    check_pure_cpp();
    check_py_child();
    check_casting();
    check_casting_without_explicit_base();
  }

  cout << "[ Done ]" << endl;

  return 0;
}
