/*
    pybind11/numpy_dtype_user.h: User-defined data types for NumPy

    Copyright (c) 2018 Eric Cousineau <eric.cousineau@tri.global>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "numpy.h"
#include "detail/inference.h"
#include "detail/numpy_ufunc.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <functional>
#include <utility>
#include <vector>
#include <typeindex>

// N.B. For NumPy dtypes, `custom` tends to mean record-like structures, while
// `user-defined` means teaching NumPy about previously opaque C structures.

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

// The following code effectively creates a separate instance system than what
// pybind11 nominally has. This is done because, at present, it's difficult to
// have pybind11 extend other python types, in this case, `np.generic` /
// `PyGenericArrType_Type` (#1170).

// TODO(eric.cousineau): Get rid of this structure if #1170 can be resolved.

// Watered down version of `detail::type_info`, specifically for
// NumPy user dtypes.
struct dtype_info {
  handle cls;
  int dtype_num{-1};
  std::map<void*, PyObject*> instance_to_py;

  // Provides mutable entry for a registered type, with option to create.
  template <typename T>
  static dtype_info& get_mutable_entry(bool is_new = false) {
    auto& internals = get_internals();
    std::type_index id(typeid(T));
    if (is_new) {
      if (internals.find(id) != internals.end())
        pybind11_fail("Class already registered");
      return internals[id];
    } else {
      return internals.at(id);
    }
  }

  // Provides immutable entry for a registered type.
  template <typename T>
  static const dtype_info& get_entry() {
    return get_internals().at(std::type_index(typeid(T)));
  }

 private:
  using internals = std::map<std::type_index, dtype_info>;
  // TODO(eric.cousineau): Store in internals.
  static internals& get_internals() {
    static internals* ptr = &get_or_create_shared_data<internals>("_numpy_dtype_user_internals");
    return *ptr;
  }
};

// Provides `PyObject`-extension, akin to `detail::instance`.
template <typename Class>
struct dtype_user_instance {
  PyObject_HEAD
  // TODO(eric.cousineau): Consider storing a unique_ptr to reduce the number
  // of temporaries.
  Class value;

  // Extracts C++ pointer from a given python object. No type checking is done.
  static Class* load_raw(PyObject* src) {
    dtype_user_instance* obj = (dtype_user_instance*)src;
    return &obj->value;
  }

  // Allocates an instance.
  static dtype_user_instance* alloc_py() {
    auto cls = dtype_info::get_entry<Class>().cls;
    PyTypeObject* cls_raw = (PyTypeObject*)cls.ptr();
    return (dtype_user_instance*)cls_raw->tp_alloc((PyTypeObject*)cls.ptr(), 0);
  }

  // Implementation for `tp_new` slot.
  static PyObject* tp_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    // N.B. `__init__` should call the in-place constructor.
    auto obj = alloc_py();
    // // Register.
    auto& entry = dtype_info::get_mutable_entry<Class>();
    entry.instance_to_py[&obj->value] = (PyObject*)obj;
    return (PyObject*)obj;
  }

  // Implementation for `tp_dealloc` slot.
  static void tp_dealloc(PyObject* self) {
    Class* value = load_raw(self);
    // Call destructor.
    value->~Class();
    // Deregister.
    auto& entry = dtype_info::get_mutable_entry<Class>();
    entry.instance_to_py.erase(value);
  }

  // Instance finding. Returns empty `object` if nothing is found.
  static object find_existing(const Class* value) {
    auto& entry = dtype_info::get_entry<Class>();
    auto it = entry.instance_to_py.find((void*)value);
    if (it == entry.instance_to_py.end())
      return {};
    else {
      return reinterpret_borrow<object>(it->second);
    }
  }
};

// Implementation of `type_caster` interface `dtype_user_instance<>`s.
template <typename Class>
struct dtype_user_caster {
  static constexpr auto name = detail::_<Class>();
  using DTypePyObject = dtype_user_instance<Class>;
  static handle cast(const Class& src, return_value_policy, handle) {
    object h = DTypePyObject::find_existing(&src);
    // TODO(eric.cousineau): Handle parenting?
    if (!h) {
      // Make new instance.
      DTypePyObject* obj = DTypePyObject::alloc_py();
      obj->value = src;
      h = reinterpret_borrow<object>((PyObject*)obj);
      return h.release();
    }
    return h.release();
  }

  static handle cast(const Class* src, return_value_policy, handle) {
    object h = DTypePyObject::find_existing(src);
    if (h) {
      return h.release();
    } else {
      throw cast_error("Cannot find existing instance");
    }
  }

  bool load(handle src, bool convert) {
    auto cls = dtype_info::get_entry<Class>().cls;
    object obj;
    if (!isinstance(src, cls)) {
      if (convert) {
        // Just try to call it.
        // TODO(eric.cousineau): Take out the Python middle man?
        // Use registered ufuncs? See how `implicitly_convertible` is
        // implemented.
        obj = cls(src);
      } else {
        return false;
      }
    } else {
      obj = reinterpret_borrow<object>(src);
    }
    ptr_ = DTypePyObject::load_raw(obj.ptr());
    return true;
  }
  // Copy `type_caster_base`.
  template <typename T_> using cast_op_type =
      pybind11::detail::cast_op_type<T_>;

  operator Class&() { return *ptr_; }
  operator Class*() { return ptr_; }
  Class* ptr_{};
};

// Ensures that `dtype_user_caster` can cast pointers. See `cast.h`.
template <typename T>
struct cast_is_known_safe<T,
    enable_if_t<std::is_base_of<
        dtype_user_caster<intrinsic_t<T>>, make_caster<T>>::value>>
    : public std::true_type {};

// Maps a pybind11 operator (using py::self) to a NumPy UFunc name.
inline const char* get_ufunc_name(detail::op_id id) {
  using namespace detail;
  static std::map<op_id, const char*> m = {
    // https://docs.scipy.org/doc/numpy/reference/routines.math.html
    {op_add, "add"},
    {op_neg, "negative"},
    {op_mul, "multiply"},
    {op_div, "divide"},
    {op_pow, "power"},
    {op_sub, "subtract"},
    // https://docs.scipy.org/doc/numpy/reference/routines.logic.htmls
    {op_gt, "greater"},
    {op_ge, "greater_equal"},
    {op_lt, "less"},
    {op_le, "less_equal"},
    {op_eq, "equal"},
    {op_ne, "not_equal"},
    {op_bool, "nonzero"},
    {op_invert, "logical_not"}
    // TODO(eric.cousineau): Add something for junction-style logic?
  };
  return m.at(id);
}

// Provides implementation of `npy_format_decsriptor` for a user-defined dtype.
template <typename Class>
struct dtype_user_npy_format_descriptor {
    static constexpr auto name = detail::_<Class>();
    static pybind11::dtype dtype() {
        int dtype_num = dtype_info::get_entry<Class>().dtype_num;
        if (auto ptr = detail::npy_api::get().PyArray_DescrFromType_(dtype_num))
            return reinterpret_borrow<pybind11::dtype>(ptr);
        pybind11_fail("Unsupported buffer format!");
    }
};

NAMESPACE_END(detail)

template <typename Class_>
class dtype_user : public class_<Class_> {
 public:
  using Base = class_<Class_>;
  using Class = Class_;
  using DTypePyObject = detail::dtype_user_instance<Class>;

  dtype_user(handle scope, const char* name) : Base(none()) {
    register_type(name);
    scope.attr(name) = self();
    auto& entry = detail::dtype_info::get_mutable_entry<Class>(true);
    entry.cls = self();
    // Registry numpy type.
    // (Note that not registering the type will result in infinte recursion).
    entry.dtype_num = register_numpy();

    // Register default ufunc cast to `object`.
    this->def_ufunc_cast([](const Class& self) { return cast(self); });
    this->def_ufunc_cast([](object self) { return cast<Class>(self); });
  }

  ~dtype_user() {
    check();    
  }

  template <typename ... Args>
  dtype_user& def(const char* name, Args&&... args) {
    base().def(name, std::forward<Args>(args)...);
    return *this;
  }

  template <typename ... Args, typename... Extra>
  dtype_user& def(detail::initimpl::constructor<Args...>&& init, const Extra&... extra) {
    // See notes in `add_init`.
    add_init([](Class* self, Args... args) {
      // Old-style. No factories for now.
      new (self) Class(std::forward<Args>(args)...);
    });
    return *this;
  }

  template <detail::op_id id, detail::op_type ot,
      typename L, typename R, typename... Extra>
  dtype_user& def_ufunc(
      const detail::op_<id, ot, L, R>& op, const Extra&... extra) {
    using op_ = detail::op_<id, ot, L, R>;
    using op_impl = typename op_::template info<dtype_user>::op;
    auto func = &op_impl::execute;
    const char* ufunc_name = get_ufunc_name(id);
    // Define operators.
    this->def(op_impl::name(), func, is_operator(), extra...);
    // Register ufunction.
    auto func_infer = detail::function_inference::run(func);
    using Func = decltype(func_infer);
    constexpr int N = Func::Args::size;
    detail::ufunc_register<Class>(
        detail::get_py_ufunc(ufunc_name), func, detail::ufunc_nargs<N>{});
    return *this;
  }

  // Nominal operator.
  template <detail::op_id id, detail::op_type ot,
      typename L, typename R, typename... Extra>
  dtype_user& def(
      const detail::op_<id, ot, L, R>& op, const Extra&... extra) {
    base().def(op, extra...);
    return *this;
  }

  template <typename Func_>
  dtype_user& def_ufunc_cast(Func_&& func) {
    auto func_infer = detail::function_inference::run(func);
    using Func = decltype(func_infer);
    using From = detail::intrinsic_t<typename Func::Args::template type_at<0>>;
    using To = detail::intrinsic_t<typename Func::Return>;
    detail::ufunc_register_cast<From, To>(func);
    return *this;
  }

 private:
  Base& base() { return *this; }
  object& self() { return *this; }
  const object& self() const { return *this; }

  void check() const {
    // This `dict` should indicate whether we've directly overridden methods.
    dict d = self().attr("__dict__");
    // Without these, numpy goes into infinite recursion. Haven't bothered to
    // figure out exactly why.
    if (!d.contains("__repr__"))
      pybind11_fail("Class is missing explicit __repr__");
    if (!d.contains("__str__"))
      pybind11_fail("Class is missing explicit __str__");
  }

  template <typename Func>
  void add_init(Func&& f) {
    // Do not construct this with the name `__init__` as `cpp_function` will
    // try to have this register the instance, and most likely segfault.
    this->def("_dtype_init", std::forward<Func>(f));
    // Ensure that this is called by a non-pybind11-instance `__init__`.
    dict d = self().attr("__dict__");
    if (!d.contains("__init__")) {
      auto init = self().attr("_dtype_init");
      auto func = cpp_function(
          [init](handle self, args args, kwargs kwargs) {
            // Dispatch.
            self.attr("_dtype_init")(*args, **kwargs);
          }, is_method(self()));
      self().attr("__init__") = func;
    }
  }

  void register_type(const char* name) {
    // Ensure we initialize NumPy before accessing `PyGenericArrType_Type`.
    auto& api = detail::npy_api::get();
    // Loosely uses https://stackoverflow.com/a/12505371/7829525 as well.
    auto heap_type = (PyHeapTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
    if (!heap_type)
        pybind11_fail("dtype_user: Could not register heap type");
    heap_type->ht_name = str(name).release().ptr();
    // It's painful to inherit from `np.generic`, because it has no `tp_new`.
    auto& ClassObject_Type = heap_type->ht_type;
    ClassObject_Type.tp_base = api.PyGenericArrType_Type_;
    ClassObject_Type.tp_new = &DTypePyObject::tp_new;
    ClassObject_Type.tp_dealloc = &DTypePyObject::tp_dealloc;
    ClassObject_Type.tp_name = name;  // Er... scope?
    ClassObject_Type.tp_basicsize = sizeof(DTypePyObject);
    ClassObject_Type.tp_getset = 0;
    ClassObject_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
    if (PyType_Ready(&ClassObject_Type) != 0)
        pybind11_fail("dtype_user: Unable to initialize class");
    self() = reinterpret_borrow<object>(handle((PyObject*)&ClassObject_Type));
  }

  int register_numpy() {
    using detail::npy_api;
    // Adapted from `numpy/core/multiarray/src/test_rational.c.src`.
    // Define NumPy description.
    auto type = (PyTypeObject*)self().ptr();
    typedef struct { char c; Class r; } align_test;
    static detail::PyArray_ArrFuncs arrfuncs;
    static detail::PyArray_Descr descr = {
        PyObject_HEAD_INIT(0)
        type,                   /* typeobj */
        'V',                    /* kind (V = arbitrary) */
        'r',                    /* type */
        '=',                    /* byteorder */
        // TODO(eric.cousineau): NPY_NEEDS_INIT?
        npy_api::NPY_NEEDS_PYAPI_ | npy_api::NPY_USE_GETITEM_ |
            npy_api::NPY_USE_SETITEM_, /* flags */
        0,                      /* type_num */
        sizeof(Class),          /* elsize */
        offsetof(align_test,r), /* alignment */
        0,                      /* subarray */
        0,                      /* fields */
        0,                      /* names */
        &arrfuncs,  /* f */
    };

    auto& api = npy_api::get();
    Py_TYPE(&descr) = api.PyArrayDescr_Type_;

    api.PyArray_InitArrFuncs_(&arrfuncs);

    using detail::npy_intp;

    // https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html
    arrfuncs.getitem = (void*)+[](void* in, void* arr) -> PyObject* {
        auto item = (const Class*)in;
        return cast(*item).release().ptr();
    };
    arrfuncs.setitem = (void*)+[](PyObject* in, void* out, void* arr) {
        detail::dtype_user_caster<Class> caster;
        if (!caster.load(in, true))
            pybind11_fail("dtype_user: Could not convert during `setitem`");
        // Cut out the middle-man?
        *(Class*)out = caster;
        return 0;
    };
    arrfuncs.copyswap = (void*)+[](void* dst, void* src, int swap, void* arr) {
        // TODO(eric.cousineau): Figure out actual purpose of this.
        if (!src) return;
        Class* r_dst = (Class*)dst;
        Class* r_src = (Class*)src;
        if (swap) {
            std::swap(*r_dst, *r_src);
        } else {
            *r_dst = *r_src;
        }
    };
    // - Test and ensure this doesn't overwrite our `equal` unfunc.
    arrfuncs.compare = (void*)+[](const void* d1, const void* d2, void* arr) {
      return 0;
    };
    arrfuncs.fill = (void*)+[](void* data_, npy_intp length, void* arr) {
      Class* data = (Class*)data_;
      Class delta = data[1] - data[0];
      Class r = data[1];
      npy_intp i;
      for (i = 2; i < length; i++) {
          r += delta;
          data[i] = r;
      }
      return 0;
    };
    arrfuncs.fillwithscalar = (void*)+[](
            void* buffer_raw, npy_intp length, void* value_raw, void* arr) {
        const Class* value = (const Class*)value_raw;
        Class* buffer = (Class*)buffer_raw;
        for (int k = 0; k < length; k++) {
            buffer[k] = *value;
        }
        return 0;
    };
    int dtype_num = api.PyArray_RegisterDataType_(&descr);
    if (dtype_num < 0)
        pybind11_fail("dtype_user: Could not register!");
    self().attr("dtype") =
        reinterpret_borrow<object>(handle((PyObject*)&descr));
    return dtype_num;
  }
};

NAMESPACE_END(PYBIND11_NAMESPACE)

// Ensures that we can (a) cast the type (semi) natively, and (b) integrate
// with NumPy functionality.
#define PYBIND11_NUMPY_DTYPE_USER(Type)  \
    namespace pybind11 { namespace detail { \
        template <> \
        struct type_caster<Type> : public dtype_user_caster<Type> {}; \
        template <> \
        struct npy_format_descriptor<Type> \
            : public dtype_user_npy_format_descriptor<Type> {}; \
    }}
