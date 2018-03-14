/*
    pybind11/detail/numpy_ufunc.h: Simple glue for Python UFuncs

    Copyright (c) 2018 Eric Cousineau <eric.cousineau@tri.global>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "../numpy.h"
#include "inference.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

// Gets a NumPy UFunc by name.
PyUFuncObject* get_py_ufunc(const char* name) {
  // TODO(eric.cousineau): Check type.
  module numpy = module::import("numpy");
  return (PyUFuncObject*)numpy.attr(name).ptr();
}

// Registers a function pointer as a UFunc, mapping types to dtype nums.
template <typename Type, typename ... Args>
void ufunc_register(
        PyUFuncObject* py_ufunc,
        PyUFuncGenericFunction func,
        void* data) {
    constexpr int N = sizeof...(Args);
    int dtype = npy_format_descriptor<Type>::dtype().num();
    int dtype_args[] = {npy_format_descriptor<Args>::dtype().num()...};
    if (N != py_ufunc->nargs)
        pybind11_fail("ufunc: Argument count mismatch");
    if (npy_api::get().PyUFunc_RegisterLoopForType_(
            py_ufunc, dtype, func, dtype_args, data) < 0)
        pybind11_fail("ufunc: Failed to regstiser ufunc");
}

template <int N>
using const_int = std::integral_constant<int, N>;

// Registers a unary UFunc given a lambda.
template <typename Type, int N = 1, typename Func = void>
void ufunc_register(PyUFuncObject* py_ufunc, Func func, const_int<1>) {
    auto info = detail::function_inference::run(func);
    using Info = decltype(info);
    using Arg0 = std::decay_t<typename Info::Args::template type_at<0>>;
    using Out = std::decay_t<typename Info::Return>;
    auto ufunc = [](
            char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
        Func& func = *(Func*)data;
        int step_0 = steps[0];
        int step_out = steps[1];
        int n = *dimensions;
        char *in_0 = args[0], *out = args[1];
        for (int k = 0; k < n; k++) {
            // TODO(eric.cousineau): Support pointers being changed.
            *(Out*)out = func(*(Arg0*)in_0);
            in_0 += step_0;
            out += step_out;
        }
    };
    // N.B. `new Func(...)` will never be destroyed.
    ufunc_register<Type, Arg0, Out>(py_ufunc, ufunc, new Func(func));
};

// Binary.
template <typename Type, int N = 2, typename Func = void>
void ufunc_register(PyUFuncObject* py_ufunc, Func func, const_int<2>) {
    auto info = detail::function_inference::run(func);
    using Info = decltype(info);
    using Arg0 = std::decay_t<typename Info::Args::template type_at<0>>;
    using Arg1 = std::decay_t<typename Info::Args::template type_at<1>>;
    using Out = std::decay_t<typename Info::Return>;
    auto ufunc = [](char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
        Func& func = *(Func*)data;
        int step_0 = steps[0];
        int step_1 = steps[1];
        int step_out = steps[2];
        int n = *dimensions;
        char *in_0 = args[0], *in_1 = args[1], *out = args[2];
        for (int k = 0; k < n; k++) {
            // TODO(eric.cousineau): Support pointers being fed in.
            *(Out*)out = func(*(Arg0*)in_0, *(Arg1*)in_1);
            in_0 += step_0;
            in_1 += step_1;
            out += step_out;
        }
    };
    // N.B. `new Func(...)` will never be destroyed.
    ufunc_register<Type, Arg0, Arg1, Out>(py_ufunc, ufunc, new Func(func));
};

template <typename From, typename To, typename Func>
void ufunc_register_cast(Func&& func, type_pack<From, To> = {}) {
  static auto cast_lambda = func;
  auto cast_func = [](
        void* from_, void* to_, npy_intp n,
        void* fromarr, void* toarr) {
      const From* from = (From*)from_;
      To* to = (To*)to_;
      for (npy_intp i = 0; i < n; i++)
          to[i] = cast_lambda(from[i]);
  };
  auto& api = npy_api::get();
  auto from = npy_format_descriptor<From>::dtype();
  int to_num = npy_format_descriptor<To>::dtype().num();
  auto* from_raw = from.ptr();
  if (api.PyArray_RegisterCastFunc_(from, to_num, cast_func) < 0)
      pybind11_fail("ufunc: Cannot register cast");
  if (api.PyArray_RegisterCanCast_(from, to_num, NPY_NOSCALAR) < 0)
      pybind11_fail("ufunc: Cannot register castability");
}

NAMESPACE_END(detail)

NAMESPACE_END(PYBIND11_NAMESPACE)
