/*
    pybind11/numpy_dtypes_user.h: User-defined data types for NumPy

    Copyright (c) 2018 Eric Cousineau <eric.cousineau@tri.global>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "numpy.h"
#include "numpy_ufunc.h"
#include "operators.h"
#include "detail/inference.h"

#include <array>
#include <map>
#include <utility>
#include <typeindex>

#if defined(PYBIND11_CPP14)

// N.B. For NumPy dtypes, `custom` tends to mean record-like structures, while
// `user-defined` means teaching NumPy about previously opaque C structures.

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

// The following code effectively creates a separate instance system than what
// pybind11 nominally has. This is done because, at present, it's difficult to
// have pybind11 extend other python types, in this case, `np.generic` /
// `PyGenericArrType_Type` (#1170).

// TODO(eric.cousineau): Get rid of this structure if #1170 can be resolved.

typedef PyObject* (*nb_conversion_t)(PyObject*);

// Watered down version of `detail::type_info`, specifically for
// NumPy user dtypes.
struct dtype_info {
  handle cls;
  int dtype_num{-1};
  std::map<void*, PyObject*> instance_to_py;
  std::vector<type_info::implicit_conversion_func> implicit_conversions;
  std::map<std::type_index, nb_conversion_t> nb_implicit_conversions;

  // Provides mutable entry for a registered type, with option to create.
  template <typename T>
  static dtype_info& get_mutable_entry(bool is_new = false) {
    auto& internals = get_mutable_internals();
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
    return get_mutable_internals().at(std::type_index(typeid(T)));
  }

  static const dtype_info& get_entry(std::type_index id) {
    return get_mutable_internals().at(id);
  }

  static const dtype_info* maybe_get_entry(std::type_index id) {
    const auto& internals = get_mutable_internals();
    auto iter = internals.find(id);
    if (iter != internals.end()) {
      return &iter->second;
    } else {
      return nullptr;
    }
  }

  static const std::type_index* find_entry(object cls) {
    auto& map = get_internals();
    for (auto& iter : map) {
      auto& entry = iter.second;
      if (cls.ptr() == entry.cls.ptr())
        return &iter.first;
    }
    return nullptr;
  }

 private:
  // Preferablly not to have to define this......
  using internals = std::map<std::type_index, dtype_info>;
  static const internals& get_internals() {
    return get_mutable_internals();
  }

  // TODO(eric.cousineau): Store in internals.
  static internals& get_mutable_internals() {
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
    auto obj = (dtype_user_instance*)cls_raw->tp_alloc((PyTypeObject*)cls.ptr(), 0);
    // Ensure we clear out the memory.
    memset(&obj->value, 0, sizeof(Class));
    return obj;
  }

  // Implementation for `tp_new` slot.
  static PyObject* tp_new(PyTypeObject* /*type*/, PyObject* /*args*/, PyObject* /*kwds*/) {
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
    void* raw = const_cast<Class*>(value);
    auto it = entry.instance_to_py.find(raw);
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

  static handle cast(const Class* src, return_value_policy policy, handle) {
    object h = DTypePyObject::find_existing(src);
    if (h) {
      return h.release();
    } else {
      if (policy == return_value_policy::automatic_reference ||
          policy == return_value_policy::reference) {
        throw cast_error("Cannot find existing instance");
      } else {
        // Copy the instance.
        DTypePyObject* obj = DTypePyObject::alloc_py();
        obj->value = *src;
        delete src;
        h = reinterpret_borrow<object>((PyObject*)obj);
        return h.release();
      }
    }
  }

  bool load(handle src, bool convert) {
    auto& entry = dtype_info::get_entry<Class>();
    auto cls = entry.cls;
    object obj;
    if (!isinstance(src, cls)) {
      // Check if it's an `np.array` with matching dtype.
      handle array = (PyObject*)npy_api::get().PyArray_Type_;
      if (isinstance(src, array)) {
        tuple shape = src.attr("shape");
        if (shape.size() == 0) {
          obj = src.attr("item")();
        }
      }
      if (!obj && convert) {
        // Try implicit conversions.
        for (auto& converter : entry.implicit_conversions) {
          auto temp = converter(src.ptr(), (PyTypeObject*)cls.ptr());
          if (temp) {
            obj = reinterpret_steal<object>(temp);
            loader_life_support::add_patient(obj);
            break;
          }
        }
      }
    } else {
      obj = reinterpret_borrow<object>(src);
    }
    if (!obj) {
      return false;
    }
    else {
      ptr_ = DTypePyObject::load_raw(obj.ptr());
      return true;
    }
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

// Maps a common Python function name to a NumPy UFunc name, or just returns
// the original name (for trigonometric functions).
inline const char* get_ufunc_name(const char* name) {
  static const std::map<std::string, const char*> m = {
    // https://docs.python.org/2.7/reference/datamodel.html#emulating-numeric-types
    // Use nominal ordering (e.g. `__add__`, not `__radd__`) as ordering will be handled
    // by ufunc registration.
    // Use Python 3 operator names (e.g. `__truediv__`)
    // https://docs.scipy.org/doc/numpy/reference/routines.math.html
    {"__add__", "add"},
    {"__neg__", "negative"},
    {"__pos__", "numpy_does_not_have_positive__pos__"},  // Cause errror.
    {"__mul__", "multiply"},
    // TODO(eric.cousineau): Figure out how to appropriately map `true_divide`
    // vs. `divide` when the output type is adjusted?
    {"__truediv__", "divide"},
    {"__pow__", "power"},
    {"__sub__", "subtract"},
    {"__abs__", "absolute"},
    // https://docs.scipy.org/doc/numpy/reference/routines.logic.html
    {"__gt__", "greater"},
    {"__ge__", "greater_equal"},
    {"__lt__", "less"},
    {"__le__", "less_equal"},
    {"__eq__", "equal"},
    {"__ne__", "not_equal"},
    {"__bool__", "nonzero"}, // Python3
    {"__nonzero__", "nonzero"},  // Python2.7
    {"__invert__", "logical_not"},
    // Are these necessary?
    {"min", "fmin"},
    {"max", "fmax"},
    // TODO(eric.cousineau): Add something for junction-style logic?
  };
  auto iter = m.find(name);
  if (iter != m.end())
    return iter->second;
  else
    return name;
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


template <typename From, typename To, typename Func>
struct dtype_conversion_t {
  Func func;
  bool allow_implicit_coercion{};
};

template <typename FuncIn>
static auto dtype_conversion_impl(
    FuncIn&& func_in, bool allow_implicit_coercion) {
  auto func_infer = detail::function_inference::run(func_in);
  using FuncInfer = decltype(func_infer);
  using From = detail::intrinsic_t<typename FuncInfer::Args::template type_at<0>>;
  using To = detail::intrinsic_t<typename FuncInfer::Return>;
  using Func = typename FuncInfer::Func;
  return dtype_conversion_t<From, To, Func>{
      std::forward<Func>(func_infer.func), allow_implicit_coercion};
}

NAMESPACE_END(detail)

/// Dtype methods which cannot be defined via a UFunc.
struct dtype_method {
  struct dot {};

  template <typename From, typename To>
  static auto explicit_conversion() {
    return detail::dtype_conversion_impl([](const From& in) -> To { return in; }, false);
  }

  template <typename Func>
  static auto explicit_conversion(Func&& func) {
    return detail::dtype_conversion_impl(std::forward<Func>(func), false);
  }

  template <typename From, typename To>
  static auto implicit_conversion() {
    return detail::dtype_conversion_impl([](const From& in) -> To { return in; }, true);
  }

  template <typename Func>
  static auto implicit_conversion(Func&& func) {
    return detail::dtype_conversion_impl(std::forward<Func>(func), true);
  }

  struct ufunc_only {};
};

/**
Defines a user-defined dtype.

Constraints:
* The type must be copy-constructible and assignable.
* The type *may* not have its constructor called; however, its memory *will* be
initialized to zero, so it's assignment should be robust against being assigned
from zero memory.
* The type *won't* always be destroyed, because NumPy does not have slots to
define this yet.
 */
// TODO(eric.cousineau): When defining operator overloads, it'd be nice if
// things like `operator==` didn't have its own implicit behavior...
template <typename Class_>
class dtype_user : public object {
 public:
  static_assert(
      !std::is_polymorphic<Class_>::value,
      "Cannot define NumPy dtypes for polymorphics classes.");

  using PyClass = class_<Class_>;
  using Class = Class_;
  using DTypePyObject = detail::dtype_user_instance<Class>;

  dtype_user(handle scope, const char* name) : cls_(none()) {
    register_type(name);
    scope.attr(name) = self();
    auto& entry = detail::dtype_info::get_mutable_entry<Class>(true);
    entry.cls = self();
    // Register numpy type.
    // (Note that not registering the type will result in infinte recursion).
    entry.dtype_num = register_numpy();

    // Register default ufunc cast to `object`.
    // N.B. Given how general this is, it should *NEVER* be implicit, as it
    // would interfere with more meaningful casts.
    // N.B. This works because `object` is defined to have the same memory
    // layout as `PyObject*`, thus can be registered in lieu of `PyObject*` -
    // this also effectively increases the refcount and releases the object.
    this->def_loop(dtype_method::explicit_conversion(
        [](const Class& self) { return pybind11::cast(self); }));
    object cls = self();
    this->def_loop(dtype_method::explicit_conversion([cls](object obj) -> Class {
      // N.B. We use the *constructor* rather than implicit conversions because
      // implicit conversions may not be sufficient when dealing with `object`
      // dtypes. As an example, a class can only explicitly cast to float, but
      // the array is constructed as `np.array([1., Class(2)])`. The inferred
      // dtype in this case will be `object`.
      if (!isinstance(obj, cls)) {
        // This will catch type mismatch errors.
        obj = cls(obj);
      }
      return obj.cast<Class>();
    }));
  }

  ~dtype_user() {
    check();
  }

  template <typename ... Args>
  dtype_user& def(const char* name, Args&&... args) {
    cls().def(name, std::forward<Args>(args)...);
    return *this;
  }

  template <typename ... Args>
  dtype_user& def(detail::initimpl::constructor<Args...>&&) {
    // See notes in `add_init`.
    // N.B. Do NOT use `Class*` as the argument, since that may incur recursion.
    add_init([](object py_self, Args... args) {
      // Old-style. No factories for now.
        Class* self = DTypePyObject::load_raw(py_self.ptr());
      new (self) Class(std::forward<Args>(args)...);
    });
    return *this;
  }

  /// Define UFunc loop operator.
  template <detail::op_id id, detail::op_type ot,
      typename L, typename R>
  dtype_user& def_loop(
      const detail::op_<id, ot, L, R>&, dtype_method::ufunc_only) {
    // Register ufunction with builtin name.
    // Use `op_l`. Mapping `__radd__` to `add` would require remapping argument
    // order, and screw that. We can just use the fact that `op_impl` is
    // generic.
    constexpr auto ot_norm = (ot == detail::op_r) ? detail::op_l : ot;
    using op_norm_ = detail::op_<id, ot_norm, L, R>;
    using op_norm_impl = typename op_norm_::template info<PyClass>::op;
    const char* ufunc_name = detail::get_ufunc_name(op_norm_impl::name());
    ufunc::get_builtin(ufunc_name).def_loop<Class>(&op_norm_impl::execute);
    if (std::string(ufunc_name) == "divide") {
      ufunc::get_builtin("true_divide").def_loop<Class>(&op_norm_impl::execute);
    }
    return *this;
  }

  /// Define Python and UFunc loop operator.
  template <detail::op_id id, detail::op_type ot,
      typename L, typename R>
  dtype_user& def_loop(const detail::op_<id, ot, L, R>& op) {
    // Define Python class operator.
    using op_ = detail::op_<id, ot, L, R>;
    using op_impl = typename op_::template info<PyClass>::op;
    this->def(op_impl::name(), &op_impl::execute, is_operator());

    // Define dtype operators.
    return def_loop(op, dtype_method::ufunc_only());
  }

  /// Defines a scalar function and overloads an existing NumPy UFunc loop,
  /// mapping to a buitlin name in `numpy`.
  template <typename Func>
  dtype_user& def_loop(const char* name, const Func& func) {
    cls().def(name, func);
    const char* ufunc_name = detail::get_ufunc_name(name);
    ufunc::get_builtin(ufunc_name).def_loop<Class>(func);
    return *this;
  }

  /// Defines a nominal operator.
  template <detail::op_id id, detail::op_type ot,
      typename L, typename R, typename... Extra>
  dtype_user& def(
      const detail::op_<id, ot, L, R>& op, const Extra&... extra) {
    cls().def(op, extra...);
    return *this;
  }

  /// Defines loop cast, and optionally permit implicit conversions.
  template <typename From, typename To, typename Func>
  dtype_user& def_loop(
      detail::dtype_conversion_t<From, To, Func> conv, dtype from = {}, dtype to = {}) {
    detail::ufunc_register_cast<From, To>(
        conv.func, conv.allow_implicit_coercion, from, to);
    // Define implicit conversion on the class.
    if (conv.allow_implicit_coercion) {
      if (std::is_same<To, Class>::value) {
        // TODO(eric.cousineau): Is this a good idea? It's quite confusing to
        // discard the function here.
        auto& entry = detail::dtype_info::get_mutable_entry<Class>();
        entry.implicit_conversions.push_back(
            detail::create_implicit_caster<From, Class>());
      } else {
        auto enabled = std::is_same<From, Class>{};
        register_nb_conversion<To>(enabled, conv.func);
      }
    }
    return *this;
  }

  /// Defines dot product.
  template <typename Defer = void>
  dtype_user& def_loop(dtype_method::dot) {
    // TODO(eric.cousineau): See if there is a way to define `dot` for an
    // algebra that is not closed under addition / multiplication (e.g.
    // symbolic variable -> symbolic expression).
    if (arrfuncs_->dotfunc)
      pybind11_fail("dtype: Cannot redefine `dot`");
    using detail::npy_intp;
    arrfuncs_->dotfunc = (void*)+[](
        void* ip0_, npy_intp is0, void* ip1_, npy_intp is1,
        void* op, npy_intp n, void* /*arr*/) {
      const char *ip0 = (char*)ip0_, *ip1 = (char*)ip1_;
      Class r{};
      for (npy_intp i = 0; i < n; i++) {
        const Class& v1 = *(const Class*)ip0;
        const Class& v2 = *(const Class*)ip1;
          r += v1 * v2;
          ip0 += is0;
          ip1 += is1;
        }
        *(Class*)op = r;
    };
    return *this;
  }

  /// Access a class_ view of the type. Please be careful when adding methods
  /// or attributes, as they may conflict with how NumPy works.
  PyClass& cls() { return cls_; }

 private:
  object& self() { return *this; }
  const object& self() const { return *this; }

  void check() const {
    auto warn = [](const std::string& msg) {
      // TODO(eric.cousineau): Figure out better warning type.
      PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 0);
    };
    // This `dict` should indicate whether we've directly overridden methods.
    dict d = self().attr("__dict__");
    // Without these, numpy goes into infinite recursion. Haven't bothered to
    // figure out exactly why.
    if (!d.contains("__repr__"))
      warn("dtype: Class is missing explicit __repr__!");
    if (!d.contains("__str__"))
      warn("dtype: Class is missing explicit __str__!");
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
            init(self, *args, **kwargs);
          }, is_method(self()));
      self().attr("__init__") = func;
    }
  }

  template <typename T>
  static PyObject* handle_nb_conversion(PyObject* from) {
    auto& entry = detail::dtype_info::get_entry<Class>();
    auto& map = entry.nb_implicit_conversions;
    // Check for available conversions.
    std::type_index id(typeid(T));
    auto iter = map.find(id);
    if (iter != map.end()) {
      return iter->second(from);
    } else {
      PyErr_SetString(
        PyExc_TypeError,
        "dtype_user: Direct casting via Python not supported");
      return nullptr;
    }
  }

  template <typename To, typename Func>
  void register_nb_conversion(std::true_type, const Func& func) {
    auto& entry = detail::dtype_info::get_mutable_entry<Class>();
    std::type_index id(typeid(To));
    auto& map = entry.nb_implicit_conversions;
    assert(map.find(id) == map.end());
    static Func func_static = func;
    detail::nb_conversion_t nb_conversion = +[](PyObject* from_py) -> PyObject* {
      Class* from = pybind11::cast<Class*>(from_py);
      To to = func_static(*from);
      return pybind11::cast<To>(to).release().ptr();
    };
    map[id] = nb_conversion;
  }

  template <typename To, typename Func>
  void register_nb_conversion(std::false_type, const Func&) {}

  static int disable_nb_coerce(PyObject**, PyObject**) {
    PyErr_SetString(
      PyExc_TypeError,
      "dtype_user: Direct coercion via Python not supported");
    return 1;
  }

  void register_type(const char* name) {
    // Ensure we initialize NumPy before accessing `PyGenericArrType_Type`.
    auto& api = detail::npy_api::get();
    // Loosely uses https://stackoverflow.com/a/12505371/7829525 as well.
    auto heap_type = (PyHeapTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
    if (!heap_type)
        pybind11_fail("dtype_user: Could not register heap type");
    heap_type->ht_name = pybind11::str(name).release().ptr();
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
    // TODO(eric.cousineau): Figure out how to catch recursions with
    // `tp_as_number` and casting, when it's not defined
    static auto tp_as_number = *ClassObject_Type.tp_as_number;
    ClassObject_Type.tp_as_number = &tp_as_number;
    // TODO(eric.cousineau): Figure out how to use more generic dispatch on
    // this object. If we use the `np.generic` stuff, we end up getting
    // recursive loops.
    tp_as_number.nb_float = &handle_nb_conversion<double>;
    tp_as_number.nb_int = &handle_nb_conversion<int>;
#if PY_VERSION_HEX < 0x03000000
    tp_as_number.nb_long = &handle_nb_conversion<int>;
    tp_as_number.nb_coerce = &disable_nb_coerce;
#endif
    // Create views into created type.
    self() = reinterpret_borrow<object>(handle((PyObject*)&ClassObject_Type));
    cls_ = self();
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
        npy_api::NPY_NEEDS_PYAPI_ | npy_api::NPY_USE_GETITEM_ |
            npy_api::NPY_USE_SETITEM_ |
            npy_api::NPY_NEEDS_INIT_, /* flags */
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
    arrfuncs.getitem = (void*)+[](void* in, void* /*arr*/) -> PyObject* {
        auto item = (const Class*)in;
        return pybind11::cast(*item).release().ptr();
    };
    arrfuncs.setitem = (void*)+[](PyObject* in, void* out, void* /*arr*/) {
        detail::loader_life_support guard{};
        detail::dtype_user_caster<Class> caster;
        if (!caster.load(in, true)) {
          PyErr_SetString(
              PyExc_TypeError,
              "dtype_user: Could not convert during `setitem`");
          return -1;
        }
        *(Class*)out = caster;
        return 0;
    };
    arrfuncs.copyswap = (void*)+[](void* dst, void* src, int swap, void* /*arr*/) {
        if (!src) return;
        Class* r_dst = (Class*)dst;
        Class* r_src = (Class*)src;
        if (swap) {
            PyErr_SetString(
                PyExc_NotImplementedError,
                "dtype_user: `swap` not implemented");
        } else {
            *r_dst = *r_src;
        }
    };
    arrfuncs.copyswapn = (void*)+[](void* dst, npy_intp dstride, void* src,
                          npy_intp sstride, npy_intp n, int swap, void*) {
        if (!src) return;
        if (swap) {
            PyErr_SetString(
                PyExc_NotImplementedError,
                "dtype_user: `swap` not implemented");
        } else {
            char* c_dst = (char*)dst;
            char* c_src = (char*)src;
            for (int k = 0; k < n; k++) {
                Class* r_dst = (Class*)c_dst;
                Class* r_src = (Class*)c_src;
                *r_dst = *r_src;
                c_dst += dstride;
                c_src += sstride;
            }
        }
    };
    // - Ensure this doesn't overwrite our `equal` unfunc.
    arrfuncs.compare = (void*)+[](const void* /*d1*/, const void* /*d2*/, void* /*arr*/) {
      pybind11_fail("dtype: `compare` should not be called for pybind11 custom dtype");
    };
    arrfuncs.fillwithscalar = (void*)+[](
            void* buffer_raw, npy_intp length, void* value_raw, void* /*arr*/) {
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
    arrfuncs_ = &arrfuncs;
    return dtype_num;
  }

  PyClass cls_;
  detail::PyArray_ArrFuncs* arrfuncs_{};
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

#endif  // defined(PYBIND11_CPP14)
