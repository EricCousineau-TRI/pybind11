/*
    pybind11/numpy_ufunc.h: Simple glue for Python UFuncs

    Copyright (c) 2018 Eric Cousineau <eric.cousineau@tri.global>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "numpy.h"
#include "detail/inference.h"

#if defined(PYBIND11_CPP14)

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

// Utilities

// Builtins registered using numpy/build/{...}/numpy/core/include/numpy/__umath_generated.c

template <typename... Args>
struct ufunc_ptr {
  PyUFuncGenericFunction func{};
  void* data{};
};

// Unary ufunc.
template <typename Arg0, typename Out, typename Func>
auto ufunc_to_ptr(Func func, type_pack<Arg0, Out>) {
    auto ufunc = [](
            char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
        Func& func = *(Func*)data;
        npy_intp step_0 = steps[0];
        npy_intp step_out = steps[1];
        npy_intp n = *dimensions;
        char *in_0 = args[0], *out = args[1];
        for (npy_intp k = 0; k < n; k++) {
            // TODO(eric.cousineau): Support pointers being changed.
            *(Out*)out = func(*(Arg0*)in_0);
            in_0 += step_0;
            out += step_out;
        }
    };
    // N.B. `new Func(...)` will never be destroyed.
    return ufunc_ptr<Arg0, Out>{ufunc, new Func(func)};
}

// Binary ufunc.
template <typename Arg0, typename Arg1, typename Out, typename Func = void>
auto ufunc_to_ptr(Func func, type_pack<Arg0, Arg1, Out>) {
    auto ufunc = [](char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
        Func& func = *(Func*)data;
        npy_intp step_0 = steps[0];
        npy_intp step_1 = steps[1];
        npy_intp step_out = steps[2];
        npy_intp n = *dimensions;
        char *in_0 = args[0], *in_1 = args[1], *out = args[2];
        for (npy_intp k = 0; k < n; k++) {
            // TODO(eric.cousineau): Support pointers being fed in.
            *(Out*)out = func(*(Arg0*)in_0, *(Arg1*)in_1);
            in_0 += step_0;
            in_1 += step_1;
            out += step_out;
        }
    };
    // N.B. `new Func(...)` will never be destroyed.
    return ufunc_ptr<Arg0, Arg1, Out>{ufunc, new Func(func)};
}

// Generic dispatch.
template <typename Func>
auto ufunc_to_ptr(Func func) {
    auto info = detail::function_inference::run(func);
    using Info = decltype(info);
    auto type_args = type_pack_apply<std::decay_t>(
        type_pack_concat(
            typename Info::Args{},
            type_pack<typename Info::Return>{}));
    return ufunc_to_ptr(func, type_args);
}

template <typename T>
void assert_ufunc_dtype_valid() {
  auto T_dtype = dtype::of<T>();
  int num = T_dtype.num();
  bool is_object = std::is_same<T, object>::value;
  if (num == npy_api::NPY_OBJECT_ && !is_object) {
    std::string message = "ufunc: Cannot handle `dtype=object` when T != `py::object` ";
    message += "(where T = " + type_id<T>() + "). ";
    message += "Please register function using `py::object`";
    pybind11_fail(message.c_str());
  }
}

template <typename From, typename To, typename Func>
void ufunc_register_cast(
    Func&& func, bool allow_coercion,
    dtype from = {}, dtype to = {}, type_pack<From, To> = {}) {
  assert_ufunc_dtype_valid<From>();
  assert_ufunc_dtype_valid<To>();
  static auto cast_lambda = detail::function_inference::run(func).func;
  auto cast_func = +[](
        void* from_, void* to_, npy_intp n,
        void* /*fromarr*/, void* /*toarr*/) {
      const From* from = (From*)from_;
      To* to = (To*)to_;
      for (npy_intp i = 0; i < n; i++)
          to[i] = cast_lambda(from[i]);
  };
  auto& api = npy_api::get();
  if (!from) {
    from = npy_format_descriptor<From>::dtype();
  }
  if (!to) {
    to = npy_format_descriptor<To>::dtype();
  }
  int to_num = to.num();
  auto from_raw = (PyArray_Descr*)from.ptr();
  if (from.num() == npy_api::NPY_OBJECT_ && !std::is_same<From, object>::value)
      pybind11_fail(
          "ufunc: Registering conversion from `dtype=object` with From != `py::object` is not supported");
  if (api.PyArray_RegisterCastFunc_(from_raw, to_num, cast_func) < 0)
      pybind11_fail("ufunc: Cannot register cast");
  if (allow_coercion) {
    if (api.PyArray_RegisterCanCast_(
            from_raw, to_num, npy_api::NPY_NOSCALAR_) < 0)
        pybind11_fail(
            "ufunc: Cannot register implicit / coercion cast capability");
  }
}

NAMESPACE_END(detail)

class ufunc : public object {
public:
    ufunc(object ptr_in) : object(ptr_in) {
        // TODO(eric.cousineau): Check type.
        if (!self() || self().is_none())
            pybind11_fail("ufunc: Cannot create from empty or None object");
        entries.reset(new entries_t(ptr()));
    }

    ufunc(detail::PyUFuncObject* ptr_in)
        : ufunc(reinterpret_borrow<object>((PyObject*)ptr_in))
    {}

    ufunc(handle scope, const char* name) : scope_{scope} {
        entries.reset(new entries_t(name));
    }

    ufunc(const ufunc&) = default;

    ~ufunc() {
        if (entries)
            finalize();
    }

    // Gets a NumPy UFunc by name.
    static ufunc get_builtin(const char* name) {
        module numpy = module::import("numpy");
        return ufunc(numpy.attr(name));
    }

    template <typename Type, typename Func>
    ufunc& def_loop(Func func_in) {
        auto func = detail::function_inference::run(func_in).func;
        do_register<Type>(detail::ufunc_to_ptr(func));
        return *this;
    }

    detail::PyUFuncObject* ptr() const {
        return (detail::PyUFuncObject*)self().ptr();
    }

    // Create UFunc object with core type functions if needed, and register user functions.
    void finalize() {
        if (!entries)
            pybind11_fail("Object already finalized");
        if (!self()) {
            // Create object and register core functions.
            auto* h = entries->create_core();
            self() = reinterpret_borrow<object>((PyObject*)h);
            scope_.attr(entries->name()) = self();
        }
        // Register user type functions.
        entries->create_user(ptr());
        // Leak memory for now so that data lives longer than UFunc.
        // TODO(eric.cousineau): Embed this in a capsule and use `keep_alive`.
        (void)new std::shared_ptr<entries_t>(entries);
    }

private:
    object& self() { return *this; }
    const object& self() const { return *this; }

    // Registers a function pointer as a UFunc, mapping types to dtype nums.
    template <typename Type, typename ... Args>
    void do_register(detail::ufunc_ptr<Args...> user) {
        constexpr int N = sizeof...(Args);
        constexpr int nin = N - 1;
        constexpr int nout = 1;
        entries->init_or_check_args(nin, nout);

        int dtype = dtype::of<Type>().num();
        int dummy[] = {(detail::assert_ufunc_dtype_valid<Args>(), 0)...};
        (void)dummy;
        std::vector<int> dtype_args = {dtype::of<Args>().num()...};
        // Determine if we need to make a new ufunc.
        bool is_core = true;
        for (int i = 0; i < N; ++i) {
            size_t ii = static_cast<size_t>(i);
            if (dtype_args[ii] >= detail::npy_api::constants::NPY_USERDEF_)
                is_core = false;
        }
        if (is_core) {
            // TODO: Consider supporting `PyUFunc_ReplaceLoopBySignature_`?
            if (self())
                pybind11_fail("ufunc: Can't add/replace signatures for core types for an existing ufunc");
            entries->queue_core(user.func, user.data, dtype_args);
        } else {
            entries->queue_user(user.func, user.data, dtype, dtype_args);
        }
    }

    // These are only used if we have something new.
    handle scope_{};

    struct entries_t {
        // Initialize from existing object.
        entries_t(detail::PyUFuncObject* h) {
            nin_ = h->nin;
            nout_ = h->nout;
        }

        // Set up to create a new instance.
        entries_t(const char* name) : name_(name) {}

        void init_or_check_args(int nin, int nout) {
            if (nin_ != -1 && nout_ != -1) {
                if (nin_ != nin)
                    pybind11_fail("ufunc: Input count mismatch");
                if (nout_ != nout)
                    pybind11_fail("ufunc: Output count mismatch");
            }
            nin_ = nin;
            nout_ = nout;
        }

        const char* name() const { return name_.c_str(); }

        void queue_core(detail::PyUFuncGenericFunction func, void* data, const std::vector<int>& dtype_args) {
            assert(nin_ != -1 && nout_ != -1);
            assert((int)dtype_args.size() == nin_ + nout_);
            // Store core functionn.
            core_funcs_.push_back(func);
            core_data_.push_back(data);
            size_t ncore = core_funcs_.size();
            size_t t_index = core_type_args_.size();
            int nargs = nin_ + nout_;
            core_type_args_.resize(ncore * static_cast<size_t>(nargs));
            for (size_t i = 0; i < dtype_args.size(); ++i) {
                core_type_args_.at(t_index++) = (char)dtype_args[i];
            }
        }

        void queue_user(detail::PyUFuncGenericFunction func, void* data, int dtype, const std::vector<int>& dtype_args) {
            assert(nin_ != -1 && nout_ != -1);
            assert((int)dtype_args.size() == nin_ + nout_);
            user_funcs_.push_back(func);
            user_data_.push_back(data);
            user_types_.push_back(dtype);
            user_type_args_.push_back(dtype_args);
        }

        detail::PyUFuncObject* create_core() {
            int ncore = (int)core_funcs_.size();
            char* name_raw = const_cast<char*>(name());
            return (detail::PyUFuncObject*)detail::npy_api::get().PyUFunc_FromFuncAndData_(
                core_funcs_.data(), core_data_.data(), core_type_args_.data(), ncore,
                nin_, nout_, detail::npy_api::constants::PyUFunc_None_, name_raw, nullptr, 0);
        }

        void create_user(detail::PyUFuncObject* h) {
            size_t nuser = user_funcs_.size();
            for (size_t i = 0; i < nuser; ++i) {
                if (detail::npy_api::get().PyUFunc_RegisterLoopForType_(
                        h, user_types_[i], user_funcs_[i], user_type_args_[i].data(), user_data_[i]) < 0)
                    pybind11_fail("ufunc: Failed to register custom ufunc");
            }
        }

    private:
        int nin_{-1};
        int nout_{-1};
        std::string name_{};

        // Core.
        std::vector<detail::PyUFuncGenericFunction> core_funcs_;
        std::vector<void*> core_data_;
        std::vector<char> core_type_args_;

        // User.
        std::vector<detail::PyUFuncGenericFunction> user_funcs_;
        std::vector<void*> user_data_;
        std::vector<int> user_types_;
        std::vector<std::vector<int>> user_type_args_;
    };

    std::shared_ptr<entries_t> entries;
};

NAMESPACE_END(PYBIND11_NAMESPACE)

#endif  // defined(PYBIND11_CPP14)
