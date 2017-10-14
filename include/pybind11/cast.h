/*
    pybind11/cast.h: Partial template specializations to cast between
    C++ and Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pytypes.h"
#include "detail/typeid.h"
#include "detail/descr.h"
#include "detail/internals.h"
#include <array>
#include <limits>
#include <tuple>

#if defined(PYBIND11_CPP17)
#  if defined(__has_include)
#    if __has_include(<string_view>)
#      define PYBIND11_HAS_STRING_VIEW
#    endif
#  elif defined(_MSC_VER)
#    define PYBIND11_HAS_STRING_VIEW
#  endif
#endif
#ifdef PYBIND11_HAS_STRING_VIEW
#include <string_view>
#endif

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

/// A life support system for temporary objects created by `type_caster::load()`.
/// Adding a patient will keep it alive up until the enclosing function returns.
class loader_life_support {
public:
    /// A new patient frame is created when a function is entered
    loader_life_support() {
        get_internals().loader_patient_stack.push_back(nullptr);
    }

    /// ... and destroyed after it returns
    ~loader_life_support() {
        auto &stack = get_internals().loader_patient_stack;
        if (stack.empty())
            pybind11_fail("loader_life_support: internal error");

        auto ptr = stack.back();
        stack.pop_back();
        Py_CLEAR(ptr);

        // A heuristic to reduce the stack's capacity (e.g. after long recursive calls)
        if (stack.capacity() > 16 && stack.size() != 0 && stack.capacity() / stack.size() > 2)
            stack.shrink_to_fit();
    }

    /// This can only be used inside a pybind11-bound function, either by `argument_loader`
    /// at argument preparation time or by `py::cast()` at execution time.
    PYBIND11_NOINLINE static void add_patient(handle h) {
        auto &stack = get_internals().loader_patient_stack;
        if (stack.empty())
            throw cast_error("When called outside a bound function, py::cast() cannot "
                             "do Python -> C++ conversions which require the creation "
                             "of temporary values");

        auto &list_ptr = stack.back();
        if (list_ptr == nullptr) {
            list_ptr = PyList_New(1);
            if (!list_ptr)
                pybind11_fail("loader_life_support: error allocating list");
            PyList_SET_ITEM(list_ptr, 0, h.inc_ref().ptr());
        } else {
            auto result = PyList_Append(list_ptr, h.ptr());
            if (result == -1)
                pybind11_fail("loader_life_support: error adding patient");
        }
    }
};

// Gets the cache entry for the given type, creating it if necessary.  The return value is the pair
// returned by emplace, i.e. an iterator for the entry and a bool set to `true` if the entry was
// just created.
inline std::pair<decltype(internals::registered_types_py)::iterator, bool> all_type_info_get_cache(PyTypeObject *type);

// Populates a just-created cache entry.
PYBIND11_NOINLINE inline void all_type_info_populate(PyTypeObject *t, std::vector<type_info *> &bases) {
    std::vector<PyTypeObject *> check;
    for (handle parent : reinterpret_borrow<tuple>(t->tp_bases))
        check.push_back((PyTypeObject *) parent.ptr());

    auto const &type_dict = get_internals().registered_types_py;
    for (size_t i = 0; i < check.size(); i++) {
        auto type = check[i];
        // Ignore Python2 old-style class super type:
        if (!PyType_Check((PyObject *) type)) continue;

        // Check `type` in the current set of registered python types:
        auto it = type_dict.find(type);
        if (it != type_dict.end()) {
            // We found a cache entry for it, so it's either pybind-registered or has pre-computed
            // pybind bases, but we have to make sure we haven't already seen the type(s) before: we
            // want to follow Python/virtual C++ rules that there should only be one instance of a
            // common base.
            for (auto *tinfo : it->second) {
                // NB: Could use a second set here, rather than doing a linear search, but since
                // having a large number of immediate pybind11-registered types seems fairly
                // unlikely, that probably isn't worthwhile.
                bool found = false;
                for (auto *known : bases) {
                    if (known == tinfo) { found = true; break; }
                }
                if (!found) bases.push_back(tinfo);
            }
        }
        else if (type->tp_bases) {
            // It's some python type, so keep follow its bases classes to look for one or more
            // registered types
            if (i + 1 == check.size()) {
                // When we're at the end, we can pop off the current element to avoid growing
                // `check` when adding just one base (which is typical--i.e. when there is no
                // multiple inheritance)
                check.pop_back();
                i--;
            }
            for (handle parent : reinterpret_borrow<tuple>(type->tp_bases))
                check.push_back((PyTypeObject *) parent.ptr());
        }
    }
}

/**
 * Extracts vector of type_info pointers of pybind-registered roots of the given Python type.  Will
 * be just 1 pybind type for the Python type of a pybind-registered class, or for any Python-side
 * derived class that uses single inheritance.  Will contain as many types as required for a Python
 * class that uses multiple inheritance to inherit (directly or indirectly) from multiple
 * pybind-registered classes.  Will be empty if neither the type nor any base classes are
 * pybind-registered.
 *
 * The value is cached for the lifetime of the Python type.
 */
inline const std::vector<detail::type_info *> &all_type_info(PyTypeObject *type) {
    auto ins = all_type_info_get_cache(type);
    if (ins.second)
        // New cache entry: populate it
        all_type_info_populate(type, ins.first->second);

    return ins.first->second;
}

/**
 * Gets a single pybind11 type info for a python type.  Returns nullptr if neither the type nor any
 * ancestors are pybind11-registered.  Throws an exception if there are multiple bases--use
 * `all_type_info` instead if you want to support multiple bases.
 */
PYBIND11_NOINLINE inline detail::type_info* get_type_info(PyTypeObject *type, bool do_throw = true) {
    auto &bases = all_type_info(type);
    if (bases.size() == 0)
        return nullptr;
    if (bases.size() > 1) {
        if (do_throw)
            pybind11_fail("pybind11::detail::get_type_info: type has multiple pybind11-registered bases");
        else
            return nullptr;
    }
    return bases.front();
}

inline detail::type_info *get_local_type_info(const std::type_index &tp) {
    auto &locals = registered_local_types_cpp();
    auto it = locals.find(tp);
    if (it != locals.end())
        return it->second;
    return nullptr;
}

inline detail::type_info *get_global_type_info(const std::type_index &tp) {
    auto &types = get_internals().registered_types_cpp;
    auto it = types.find(tp);
    if (it != types.end())
        return it->second;
    return nullptr;
}

/// Return the type info for a given C++ type; on lookup failure can either throw or return nullptr.
PYBIND11_NOINLINE inline detail::type_info *get_type_info(const std::type_index &tp,
                                                          bool throw_if_missing = false) {
    if (auto ltype = get_local_type_info(tp))
        return ltype;
    if (auto gtype = get_global_type_info(tp))
        return gtype;

    if (throw_if_missing) {
        std::string tname = tp.name();
        detail::clean_type_id(tname);
        pybind11_fail("pybind11::detail::get_type_info: unable to find type info for \"" + tname + "\"");
    }
    return nullptr;
}

PYBIND11_NOINLINE inline handle get_type_handle(const std::type_info &tp, bool throw_if_missing) {
    detail::type_info *type_info = get_type_info(tp, throw_if_missing);
    return handle(type_info ? ((PyObject *) type_info->type) : nullptr);
}

struct value_and_holder {
    instance *inst;
    size_t index;
    const detail::type_info *type;
    void **vh;

    // Main constructor for a found value/holder:
    value_and_holder(instance *i, const detail::type_info *type, size_t vpos, size_t index) :
        inst{i}, index{index}, type{type},
        vh{inst->simple_layout ? inst->simple_value_holder : &inst->nonsimple.values_and_holders[vpos]}
    {}

    // Default constructor (used to signal a value-and-holder not found by get_value_and_holder())
    value_and_holder() : inst{nullptr} {}

    // Used for past-the-end iterator
    value_and_holder(size_t index) : index{index} {}

    template <typename V = void> V *&value_ptr() const {
        return reinterpret_cast<V *&>(vh[0]);
    }
    // True if this `value_and_holder` has a non-null value pointer
    explicit operator bool() const { return value_ptr(); }

    template <typename H> H &holder() const {
        return reinterpret_cast<H &>(vh[1]);
    }
    void* holder_ptr() const { return &vh[1]; }
    bool holder_constructed() const {
        return inst->simple_layout
            ? inst->simple_holder_constructed
            : inst->nonsimple.status[index] & instance::status_holder_constructed;
    }
    void set_holder_constructed(bool v = true) {
        if (inst->simple_layout)
            inst->simple_holder_constructed = v;
        else if (v)
            inst->nonsimple.status[index] |= instance::status_holder_constructed;
        else
            inst->nonsimple.status[index] &= (uint8_t) ~instance::status_holder_constructed;
    }
    bool instance_registered() const {
        return inst->simple_layout
            ? inst->simple_instance_registered
            : inst->nonsimple.status[index] & instance::status_instance_registered;
    }
    void set_instance_registered(bool v = true) {
        if (inst->simple_layout)
            inst->simple_instance_registered = v;
        else if (v)
            inst->nonsimple.status[index] |= instance::status_instance_registered;
        else
            inst->nonsimple.status[index] &= (uint8_t) ~instance::status_instance_registered;
    }
};

// Container for accessing and iterating over an instance's values/holders
struct values_and_holders {
private:
    instance *inst;
    using type_vec = std::vector<detail::type_info *>;
    const type_vec &tinfo;

public:
    values_and_holders(instance *inst) : inst{inst}, tinfo(all_type_info(Py_TYPE(inst))) {}

    struct iterator {
    private:
        instance *inst;
        const type_vec *types;
        value_and_holder curr;
        friend struct values_and_holders;
        iterator(instance *inst, const type_vec *tinfo)
            : inst{inst}, types{tinfo},
            curr(inst /* instance */,
                 types->empty() ? nullptr : (*types)[0] /* type info */,
                 0, /* vpos: (non-simple types only): the first vptr comes first */
                 0 /* index */)
        {}
        // Past-the-end iterator:
        iterator(size_t end) : curr(end) {}
    public:
        bool operator==(const iterator &other) { return curr.index == other.curr.index; }
        bool operator!=(const iterator &other) { return curr.index != other.curr.index; }
        iterator &operator++() {
            if (!inst->simple_layout)
                curr.vh += 1 + (*types)[curr.index]->holder_size_in_ptrs;
            ++curr.index;
            curr.type = curr.index < types->size() ? (*types)[curr.index] : nullptr;
            return *this;
        }
        value_and_holder &operator*() { return curr; }
        value_and_holder *operator->() { return &curr; }
    };

    iterator begin() { return iterator(inst, &tinfo); }
    iterator end() { return iterator(tinfo.size()); }

    iterator find(const type_info *find_type) {
        auto it = begin(), endit = end();
        while (it != endit && it->type != find_type) ++it;
        return it;
    }

    size_t size() { return tinfo.size(); }
};

/**
 * Extracts C++ value and holder pointer references from an instance (which may contain multiple
 * values/holders for python-side multiple inheritance) that match the given type.  Throws an error
 * if the given type (or ValueType, if omitted) is not a pybind11 base of the given instance.  If
 * `find_type` is omitted (or explicitly specified as nullptr) the first value/holder are returned,
 * regardless of type (and the resulting .type will be nullptr).
 *
 * The returned object should be short-lived: in particular, it must not outlive the called-upon
 * instance.
 */
PYBIND11_NOINLINE inline value_and_holder instance::get_value_and_holder(const type_info *find_type /*= nullptr default in common.h*/, bool throw_if_missing /*= true in common.h*/) {
    // Optimize common case:
    if (!find_type || Py_TYPE(this) == find_type->type)
        return value_and_holder(this, find_type, 0, 0);

    detail::values_and_holders vhs(this);
    auto it = vhs.find(find_type);
    if (it != vhs.end())
        return *it;

    if (!throw_if_missing)
        return value_and_holder();

#if defined(NDEBUG)
    pybind11_fail("pybind11::detail::instance::get_value_and_holder: "
            "type is not a pybind11 base of the given instance "
            "(compile in debug mode for type details)");
#else
    pybind11_fail("pybind11::detail::instance::get_value_and_holder: `" +
            std::string(find_type->type->tp_name) + "' is not a pybind11 base of the given `" +
            std::string(Py_TYPE(this)->tp_name) + "' instance");
#endif
}

PYBIND11_NOINLINE inline void instance::allocate_layout() {
    auto &tinfo = all_type_info(Py_TYPE(this));

    const size_t n_types = tinfo.size();

    if (n_types == 0)
        pybind11_fail("instance allocation failed: new instance has no pybind11-registered base types");

    simple_layout =
        n_types == 1 && tinfo.front()->holder_size_in_ptrs <= instance_simple_holder_in_ptrs();

    // Simple path: no python-side multiple inheritance, and a small-enough holder
    if (simple_layout) {
        simple_value_holder[0] = nullptr;
        simple_holder_constructed = false;
        simple_instance_registered = false;
    }
    else { // multiple base types or a too-large holder
        // Allocate space to hold: [v1*][h1][v2*][h2]...[bb...] where [vN*] is a value pointer,
        // [hN] is the (uninitialized) holder instance for value N, and [bb...] is a set of bool
        // values that tracks whether each associated holder has been initialized.  Each [block] is
        // padded, if necessary, to an integer multiple of sizeof(void *).
        size_t space = 0;
        for (auto t : tinfo) {
            space += 1; // value pointer
            space += t->holder_size_in_ptrs; // holder instance
        }
        size_t flags_at = space;
        space += size_in_ptrs(n_types); // status bytes (holder_constructed and instance_registered)

        // Allocate space for flags, values, and holders, and initialize it to 0 (flags and values,
        // in particular, need to be 0).  Use Python's memory allocation functions: in Python 3.6
        // they default to using pymalloc, which is designed to be efficient for small allocations
        // like the one we're doing here; in earlier versions (and for larger allocations) they are
        // just wrappers around malloc.
#if PY_VERSION_HEX >= 0x03050000
        nonsimple.values_and_holders = (void **) PyMem_Calloc(space, sizeof(void *));
        if (!nonsimple.values_and_holders) throw std::bad_alloc();
#else
        nonsimple.values_and_holders = (void **) PyMem_New(void *, space);
        if (!nonsimple.values_and_holders) throw std::bad_alloc();
        std::memset(nonsimple.values_and_holders, 0, space * sizeof(void *));
#endif
        nonsimple.status = reinterpret_cast<uint8_t *>(&nonsimple.values_and_holders[flags_at]);
    }
    owned = true;
}

PYBIND11_NOINLINE inline void instance::deallocate_layout() {
    if (!simple_layout)
        PyMem_Free(nonsimple.values_and_holders);
}

PYBIND11_NOINLINE inline bool isinstance_generic(handle obj, const std::type_info &tp) {
    handle type = detail::get_type_handle(tp, false);
    if (!type)
        return false;
    return isinstance(obj, type);
}

PYBIND11_NOINLINE inline std::string error_string() {
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown internal error occurred");
        return "Unknown internal error occurred";
    }

    error_scope scope; // Preserve error state

    std::string errorString;
    if (scope.type) {
        errorString += handle(scope.type).attr("__name__").cast<std::string>();
        errorString += ": ";
    }
    if (scope.value)
        errorString += (std::string) str(scope.value);

    PyErr_NormalizeException(&scope.type, &scope.value, &scope.trace);

#if PY_MAJOR_VERSION >= 3
    if (scope.trace != nullptr)
        PyException_SetTraceback(scope.value, scope.trace);
#endif

#if !defined(PYPY_VERSION)
    if (scope.trace) {
        PyTracebackObject *trace = (PyTracebackObject *) scope.trace;

        /* Get the deepest trace possible */
        while (trace->tb_next)
            trace = trace->tb_next;

        PyFrameObject *frame = trace->tb_frame;
        errorString += "\n\nAt:\n";
        while (frame) {
            int lineno = PyFrame_GetLineNumber(frame);
            errorString +=
                "  " + handle(frame->f_code->co_filename).cast<std::string>() +
                "(" + std::to_string(lineno) + "): " +
                handle(frame->f_code->co_name).cast<std::string>() + "\n";
            frame = frame->f_back;
        }
    }
#endif

    return errorString;
}

PYBIND11_NOINLINE inline handle get_object_handle(const void *ptr, const detail::type_info *type ) {
    auto &instances = get_internals().registered_instances;
    auto range = instances.equal_range(ptr);
    for (auto it = range.first; it != range.second; ++it) {
        for (auto vh : values_and_holders(it->second)) {
            if (vh.type == type)
                return handle((PyObject *) it->second);
        }
    }
    return handle();
}

inline PyThreadState *get_thread_state_unchecked() {
#if defined(PYPY_VERSION)
    return PyThreadState_GET();
#elif PY_VERSION_HEX < 0x03000000
    return _PyThreadState_Current;
#elif PY_VERSION_HEX < 0x03050000
    return (PyThreadState*) _Py_atomic_load_relaxed(&_PyThreadState_Current);
#elif PY_VERSION_HEX < 0x03050200
    return (PyThreadState*) _PyThreadState_Current.value;
#else
    return _PyThreadState_UncheckedGet();
#endif
}

// Forward declarations
inline void keep_alive_impl(handle nurse, handle patient);
inline PyObject *make_new_instance(PyTypeObject *type);

enum class LoadType {
  PureCpp,
  DerivedCppSinglePySingle,
  DerivedCppSinglePyMulti,
  DerivedCppMulti,
  /// Polymorphic casting or copy-based casting may be necessary.
  ConversionNeeded,
};

typedef type_info* base_ptr_t;
typedef const std::vector<base_ptr_t> bases_t;

inline LoadType determine_load_type(handle src, const type_info* typeinfo,
                             const bases_t** out_bases = nullptr,
                             base_ptr_t* out_base = nullptr) {
    // Null out inputs.
    if (out_bases)
        *out_bases = nullptr;
    if (out_base)
        *out_base = nullptr;
    PyTypeObject *srctype = Py_TYPE(src.ptr());
    // See `type_caster_generic::load_impl` below for more detail on comments.

    // Case 1: If src is an exact type match for the target type then we can reinterpret_cast
    // the instance's value pointer to the target type:
    if (srctype == typeinfo->type) {
        // TODO(eric.cousineau): Determine if the type is upcast from a type, which is
        // still a pure C++ object?
        return LoadType::PureCpp;
    }
    // Case 2: We have a derived class
    else if (PyType_IsSubtype(srctype, typeinfo->type)) {
        const bases_t& bases = all_type_info(srctype);
        if (out_bases)
            *out_bases = &bases;  // Copy to output for caching.
        const bool no_cpp_mi = typeinfo->simple_type;
        // Case 2a: the python type is a Python-inherited derived class that inherits from just
        // one simple (no MI) pybind11 class, or is an exact match, so the C++ instance is of
        // the right type and we can use reinterpret_cast.
        // (This is essentially the same as case 2b, but because not using multiple inheritance
        // is extremely common, we handle it specially to avoid the loop iterator and type
        // pointer lookup overhead)
        // TODO(eric.cousineau): This seems to also capture C++-registered classes as well, not just Python-derived
        // classes.
        if (bases.size() == 1 && (no_cpp_mi || bases.front()->type == typeinfo->type)) {
            return LoadType::DerivedCppSinglePySingle;
        }
        // Case 2b: the python type inherits from multiple C++ bases.  Check the bases to see if
        // we can find an exact match (or, for a simple C++ type, an inherited match); if so, we
        // can safely reinterpret_cast to the relevant pointer.
        else if (bases.size() > 1) {
           for (auto base : bases) {
               if (no_cpp_mi ? PyType_IsSubtype(base->type, typeinfo->type) : base->type == typeinfo->type) {
                   if (out_base) {
                       *out_base = base;
                   }
                   return LoadType::DerivedCppSinglePyMulti;
               }
           }
        }
        // Case 2c: C++ multiple inheritance is involved and we couldn't find an exact type match
        // in the registered bases, above, so try implicit casting (needed for proper C++ casting
        // when MI is involved).
        return LoadType::DerivedCppMulti;
    } else {
        return LoadType::ConversionNeeded;
    }
}

class type_caster_generic {
public:
    PYBIND11_NOINLINE type_caster_generic(const std::type_info &type_info)
        : typeinfo(get_type_info(type_info)), cpptype(&type_info) { }

    type_caster_generic(const type_info *typeinfo)
        : typeinfo(typeinfo), cpptype(typeinfo ? typeinfo->cpptype : nullptr) { }

    bool load(handle src, bool convert) {
        return load_impl<type_caster_generic>(src, convert);
    }

    PYBIND11_NOINLINE static handle cast(const void *_src, return_value_policy policy, handle parent,
                                         const detail::type_info *tinfo,
                                         void *(*copy_constructor)(const void *),
                                         void *(*move_constructor)(const void *),
                                         holder_erased existing_holder = {}) {
        if (!tinfo) // no type info: error will be set already
            return handle();

        void *src = const_cast<void *>(_src);
        if (src == nullptr)
            return none().release();

        const bool take_ownership = policy == return_value_policy::automatic || policy == return_value_policy::take_ownership;
        // We only come across `!existing_holder` if we are coming from `cast` and not `cast_holder`.
        const bool is_bare_ptr = !existing_holder.ptr() && existing_holder.type_id() == HolderTypeId::Unknown;

        auto it_instances = get_internals().registered_instances.equal_range(src);
        for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
            for (auto instance_type : detail::all_type_info(Py_TYPE(it_i->second))) {
                if (instance_type && same_type(*instance_type->cpptype, *tinfo->cpptype)) {
                    instance* const inst = it_i->second;

                    bool try_to_reclaim = false;
                    if (!is_bare_ptr) {
                        switch (instance_type->release_info.holder_type_id) {
                            case detail::HolderTypeId::UniquePtr: {
                                try_to_reclaim = take_ownership;
                                break;
                            }
                            case detail::HolderTypeId::SharedPtr: {
                                if (take_ownership) {
                                    // Only try to reclaim the object if (a) it is not owned and (b) has no holder.
                                    if (!inst->simple_holder_constructed) {
                                        if (inst->owned)
                                            throw std::runtime_error("Internal error?");
                                        try_to_reclaim = true;
                                    }
                                }
                                break;
                            }
                            default: {
                                // Otherwise, do not try any reclaiming.
                                break;
                            }
                        }
                    }
                    if (try_to_reclaim) {
                        // If this object has already been registered, but we wish to take ownership of it,
                        // then use the `has_cpp_release` mechanisms to reclaim ownership.
                        // @note This should be the sole occurrence of this registered object when releasing back.
                        // @note This code path should not be invoked for pure C++

                        // TODO(eric.cousineau): This may be still be desirable if this is a raw pointer...
                        // Need to think of a desirable workflow - and if there is possible interop.
                        if (!existing_holder) {
                            throw std::runtime_error("Internal error: Should have non-null holder.");
                        }
                        // TODO(eric.cousineau): This field may not be necessary if the lowest-level type is valid.
                        // See `move_only_holder_caster::load_value`.
                        if (!inst->reclaim_from_cpp) {
                            throw std::runtime_error("Instance is registered but does not have a registered reclaim method. Internal error?");
                        }
                        return inst->reclaim_from_cpp(inst, existing_holder).release();
                    } else {
                        // TODO(eric.cousineau): Should really check that ownership is consistent.
                        // e.g. if we say to take ownership of a pointer that is passed, does not have a holder...
                        // In the end, pybind11 would let ownership slip, and leak memory, possibly violating RAII (if someone is using that...)
                        return handle((PyObject *) it_i->second).inc_ref();
                    }
                }
            }
        }

        auto inst = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto wrapper = reinterpret_cast<instance *>(inst.ptr());
        wrapper->owned = false;
        void *&valueptr = values_and_holders(wrapper).begin()->value_ptr();

        switch (policy) {
            case return_value_policy::automatic:
            case return_value_policy::take_ownership:
                valueptr = src;
                wrapper->owned = true;
                break;

            case return_value_policy::automatic_reference:
            case return_value_policy::reference:
                valueptr = src;
                wrapper->owned = false;
                break;

            case return_value_policy::copy:
                if (copy_constructor)
                    valueptr = copy_constructor(src);
                else
                    throw cast_error("return_value_policy = copy, but the "
                                     "object is non-copyable!");
                wrapper->owned = true;
                break;

            case return_value_policy::move:
                if (move_constructor)
                    valueptr = move_constructor(src);
                else if (copy_constructor)
                    valueptr = copy_constructor(src);
                else
                    throw cast_error("return_value_policy = move, but the "
                                     "object is neither movable nor copyable!");
                wrapper->owned = true;
                break;

            case return_value_policy::reference_internal:
                valueptr = src;
                wrapper->owned = false;
                keep_alive_impl(inst, parent);
                break;

            default:
                throw cast_error("unhandled return_value_policy: should not happen!");
        }

        // TODO(eric.cousineau): Propagate `holder_erased` through this chain.
        tinfo->init_instance(wrapper, existing_holder.ptr());

        return inst.release();
    }

    // Base methods for generic caster; there are overridden in copyable_holder_caster
    void load_value(value_and_holder &&v_h, LoadType) {
        auto *&vptr = v_h.value_ptr();
        // Lazy allocation for unallocated values:
        if (vptr == nullptr) {
            auto *type = v_h.type ? v_h.type : typeinfo;
            vptr = type->operator_new(type->type_size);
        }
        value = vptr;
    }
    bool try_implicit_casts(handle src, bool convert) {
        for (auto &cast : typeinfo->implicit_casts) {
            type_caster_generic sub_caster(*cast.first);
            if (sub_caster.load(src, convert)) {
                value = cast.second(sub_caster.value);
                return true;
            }
        }
        return false;
    }
    bool try_direct_conversions(handle src) {
        for (auto &converter : *typeinfo->direct_conversions) {
            if (converter(src.ptr(), value))
                return true;
        }
        return false;
    }
    void check_holder_compat() {}

    PYBIND11_NOINLINE static void *local_load(PyObject *src, const type_info *ti) {
        auto caster = type_caster_generic(ti);
        if (caster.load(src, false))
            return caster.value;
        return nullptr;
    }

    /// Try to load with foreign typeinfo, if available. Used when there is no
    /// native typeinfo, or when the native one wasn't able to produce a value.
    PYBIND11_NOINLINE bool try_load_foreign_module_local(handle src) {
        constexpr auto *local_key = PYBIND11_MODULE_LOCAL_ID;
        const auto pytype = src.get_type();
        if (!hasattr(pytype, local_key))
            return false;

        type_info *foreign_typeinfo = reinterpret_borrow<capsule>(getattr(pytype, local_key));
        // Only consider this foreign loader if actually foreign and is a loader of the correct cpp type
        if (foreign_typeinfo->module_local_load == &local_load
            || (cpptype && !same_type(*cpptype, *foreign_typeinfo->cpptype)))
            return false;

        if (auto result = foreign_typeinfo->module_local_load(src.ptr(), foreign_typeinfo)) {
            value = result;
            return true;
        }
        return false;
    }

    // Implementation of `load`; this takes the type of `this` so that it can dispatch the relevant
    // bits of code between here and copyable_holder_caster where the two classes need different
    // logic (without having to resort to virtual inheritance).
    template <typename ThisT>
    PYBIND11_NOINLINE bool load_impl(handle src, bool convert) {
        if (!src) return false;
        if (!typeinfo) return try_load_foreign_module_local(src);
        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) return false;
            value = nullptr;
            return true;
        }

        auto &this_ = static_cast<ThisT &>(*this);
        this_.check_holder_compat();

        const bases_t* bases = nullptr;
        base_ptr_t base_py_multi = nullptr;
        LoadType load_type = determine_load_type(src, typeinfo, &bases, &base_py_multi);
        switch (load_type) {
            case LoadType::PureCpp: {
                this_.load_value(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder(),
                                 load_type);
                return true;
            }
            case LoadType::DerivedCppSinglePySingle: {
                this_.load_value(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder(),
                                 load_type);
                return true;
            }
            case LoadType::DerivedCppSinglePyMulti: {
                this_.load_value(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder(base_py_multi),
                                 load_type);
                return true;
            }
            case LoadType::DerivedCppMulti: {
                if (this_.try_implicit_casts(src, convert))
                    return true;
            }
            case LoadType::ConversionNeeded: {
                break;
            }
        }

        // If nothing else succeeds, perform an implicit conversion
        if (convert) {
            for (auto &converter : typeinfo->implicit_conversions) {
                auto temp = reinterpret_steal<object>(converter(src.ptr(), typeinfo->type));
                if (load_impl<ThisT>(temp, false)) {
                    loader_life_support::add_patient(temp);
                    return true;
                }
            }
            if (this_.try_direct_conversions(src))
                return true;
        }

        // Failed to match local typeinfo. Try again with global.
        if (typeinfo->module_local) {
            if (auto gtype = get_global_type_info(*typeinfo->cpptype)) {
                typeinfo = gtype;
                return load(src, false);
            }
        }

        // Global typeinfo has precedence over foreign module_local
        return try_load_foreign_module_local(src);
    }


    // Called to do type lookup and wrap the pointer and type in a pair when a dynamic_cast
    // isn't needed or can't be used.  If the type is unknown, sets the error and returns a pair
    // with .second = nullptr.  (p.first = nullptr is not an error: it becomes None).
    PYBIND11_NOINLINE static std::pair<const void *, const type_info *> src_and_type(
            const void *src, const std::type_info &cast_type, const std::type_info *rtti_type = nullptr) {
        if (auto *tpi = get_type_info(cast_type))
            return {src, const_cast<const type_info *>(tpi)};

        // Not found, set error:
        std::string tname = rtti_type ? rtti_type->name() : cast_type.name();
        detail::clean_type_id(tname);
        std::string msg = "Unregistered type : " + tname;
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        return {nullptr, nullptr};
    }

    const type_info *typeinfo = nullptr;
    const std::type_info *cpptype = nullptr;
    void *value = nullptr;
};

/**
 * Determine suitable casting operator for pointer-or-lvalue-casting type casters.  The type caster
 * needs to provide `operator T*()` and `operator T&()` operators.
 *
 * If the type supports moving the value away via an `operator T&&() &&` method, it should use
 * `movable_cast_op_type` instead.
 */
template <typename T>
using cast_op_type =
    conditional_t<std::is_pointer<remove_reference_t<T>>::value,
        typename std::add_pointer<intrinsic_t<T>>::type,
        typename std::add_lvalue_reference<intrinsic_t<T>>::type>;

/**
 * Determine suitable casting operator for a type caster with a movable value.  Such a type caster
 * needs to provide `operator T*()`, `operator T&()`, and `operator T&&() &&`.  The latter will be
 * called in appropriate contexts where the value can be moved rather than copied.
 *
 * These operator are automatically provided when using the PYBIND11_TYPE_CASTER macro.
 */
template <typename T>
using movable_cast_op_type =
    conditional_t<std::is_pointer<typename std::remove_reference<T>::type>::value,
        typename std::add_pointer<intrinsic_t<T>>::type,
    conditional_t<std::is_rvalue_reference<T>::value,
        typename std::add_rvalue_reference<intrinsic_t<T>>::type,
        typename std::add_lvalue_reference<intrinsic_t<T>>::type>>;

// std::is_copy_constructible isn't quite enough: it lets std::vector<T> (and similar) through when
// T is non-copyable, but code containing such a copy constructor fails to actually compile.
template <typename T, typename SFINAE = void> struct is_copy_constructible : std::is_copy_constructible<T> {};

// Specialization for types that appear to be copy constructible but also look like stl containers
// (we specifically check for: has `value_type` and `reference` with `reference = value_type&`): if
// so, copy constructability depends on whether the value_type is copy constructible.
template <typename Container> struct is_copy_constructible<Container, enable_if_t<all_of<
        std::is_copy_constructible<Container>,
        std::is_same<typename Container::value_type &, typename Container::reference>
    >::value>> : is_copy_constructible<typename Container::value_type> {};

#if !defined(PYBIND11_CPP17)
// Likewise for std::pair before C++17 (which mandates that the copy constructor not exist when the
// two types aren't themselves copy constructible).
template <typename T1, typename T2> struct is_copy_constructible<std::pair<T1, T2>>
    : all_of<is_copy_constructible<T1>, is_copy_constructible<T2>> {};
#endif

/// Generic type caster for objects stored on the heap
template <typename type> class type_caster_base : public type_caster_generic {
    using itype = intrinsic_t<type>;
public:
    static PYBIND11_DESCR name() { return type_descr(_<type>()); }

    type_caster_base() : type_caster_base(typeid(type)) { }
    explicit type_caster_base(const std::type_info &info) : type_caster_generic(info) { }

    static handle cast(const itype &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast(&src, policy, parent);
    }

    static handle cast(itype &&src, return_value_policy, handle parent) {
        return cast(&src, return_value_policy::move, parent);
    }

    // Returns a (pointer, type_info) pair taking care of necessary RTTI type lookup for a
    // polymorphic type.  If the instance isn't derived, returns the non-RTTI base version.
    template <typename T = itype, enable_if_t<std::is_polymorphic<T>::value, int> = 0>
    static std::pair<const void *, const type_info *> src_and_type(const itype *src) {
        const void *vsrc = src;
        auto &cast_type = typeid(itype);
        const std::type_info *instance_type = nullptr;
        if (vsrc) {
            instance_type = &typeid(*src);
            if (!same_type(cast_type, *instance_type)) {
                // This is a base pointer to a derived type; if it is a pybind11-registered type, we
                // can get the correct derived pointer (which may be != base pointer) by a
                // dynamic_cast to most derived type:
                if (auto *tpi = get_type_info(*instance_type))
                    return {dynamic_cast<const void *>(src), const_cast<const type_info *>(tpi)};
            }
        }
        // Otherwise we have either a nullptr, an `itype` pointer, or an unknown derived pointer, so
        // don't do a cast
        return type_caster_generic::src_and_type(vsrc, cast_type, instance_type);
    }

    // Non-polymorphic type, so no dynamic casting; just call the generic version directly
    template <typename T = itype, enable_if_t<!std::is_polymorphic<T>::value, int> = 0>
    static std::pair<const void *, const type_info *> src_and_type(const itype *src) {
        return type_caster_generic::src_and_type(src, typeid(itype));
    }

    static handle cast(const itype *src, return_value_policy policy, handle parent) {
        auto st = src_and_type(src);
        return type_caster_generic::cast(
            st.first, policy, parent, st.second,
            make_copy_constructor(src), make_move_constructor(src));
    }

    static handle cast_holder(const itype *src, holder_erased holder) {
        auto st = src_and_type(src);
        if (!holder) {
            throw std::runtime_error("Internal error: Should not have null holder");
        }
        return type_caster_generic::cast(
            st.first, return_value_policy::take_ownership, {}, st.second,
            nullptr, nullptr, holder);
    }

    template <typename T> using cast_op_type = cast_op_type<T>;

    operator itype*() { return (type *) value; }
    operator itype&() { if (!value) throw reference_cast_error(); return *((itype *) value); }

protected:
    using Constructor = void *(*)(const void *);

    /* Only enabled when the types are {copy,move}-constructible *and* when the type
       does not have a private operator new implementation. */
    template <typename T, typename = enable_if_t<is_copy_constructible<T>::value>>
    static auto make_copy_constructor(const T *x) -> decltype(new T(*x), Constructor{}) {
        return [](const void *arg) -> void * {
            return new T(*reinterpret_cast<const T *>(arg));
        };
    }

    template <typename T, typename = enable_if_t<std::is_move_constructible<T>::value>>
    static auto make_move_constructor(const T *x) -> decltype(new T(std::move(*const_cast<T *>(x))), Constructor{}) {
        return [](const void *arg) -> void * {
            return new T(std::move(*const_cast<T *>(reinterpret_cast<const T *>(arg))));
        };
    }

    static Constructor make_copy_constructor(...) { return nullptr; }
    static Constructor make_move_constructor(...) { return nullptr; }
};

template <typename type, typename SFINAE = void> class type_caster : public type_caster_base<type> { };
template <typename type> using make_caster = type_caster<intrinsic_t<type>>;

// Shortcut for calling a caster's `cast_op_type` cast operator for casting a type_caster to a T
template <typename T> typename make_caster<T>::template cast_op_type<T> cast_op(make_caster<T> &caster) {
    return caster.operator typename make_caster<T>::template cast_op_type<T>();
}
template <typename T> typename make_caster<T>::template cast_op_type<typename std::add_rvalue_reference<T>::type>
cast_op(make_caster<T> &&caster) {
    return std::move(caster).operator
        typename make_caster<T>::template cast_op_type<typename std::add_rvalue_reference<T>::type>();
}

template <typename type> class type_caster<std::reference_wrapper<type>> {
private:
    using caster_t = make_caster<type>;
    caster_t subcaster;
    using subcaster_cast_op_type = typename caster_t::template cast_op_type<type>;
    static_assert(std::is_same<typename std::remove_const<type>::type &, subcaster_cast_op_type>::value,
            "std::reference_wrapper<T> caster requires T to have a caster with an `T &` operator");
public:
    bool load(handle src, bool convert) { return subcaster.load(src, convert); }
    static PYBIND11_DESCR name() { return caster_t::name(); }
    static handle cast(const std::reference_wrapper<type> &src, return_value_policy policy, handle parent) {
        // It is definitely wrong to take ownership of this pointer, so mask that rvp
        if (policy == return_value_policy::take_ownership || policy == return_value_policy::automatic)
            policy = return_value_policy::automatic_reference;
        return caster_t::cast(&src.get(), policy, parent);
    }
    template <typename T> using cast_op_type = std::reference_wrapper<type>;
    operator std::reference_wrapper<type>() { return subcaster.operator subcaster_cast_op_type&(); }
};

#define PYBIND11_TYPE_CASTER(type, py_name) \
    protected: \
        type value; \
    public: \
        static PYBIND11_DESCR name() { return type_descr(py_name); } \
        template <typename T_, enable_if_t<std::is_same<type, remove_cv_t<T_>>::value, int> = 0> \
        static handle cast(T_ *src, return_value_policy policy, handle parent) { \
            if (!src) return none().release(); \
            if (policy == return_value_policy::take_ownership) { \
                auto h = cast(std::move(*src), policy, parent); delete src; return h; \
            } else { \
                return cast(*src, policy, parent); \
            } \
        } \
        operator type*() { return &value; } \
        operator type&() { return value; } \
        operator type&&() && { return std::move(value); } \
        template <typename T_> using cast_op_type = pybind11::detail::movable_cast_op_type<T_>


template <typename CharT> using is_std_char_type = any_of<
    std::is_same<CharT, char>, /* std::string */
    std::is_same<CharT, char16_t>, /* std::u16string */
    std::is_same<CharT, char32_t>, /* std::u32string */
    std::is_same<CharT, wchar_t> /* std::wstring */
>;

template <typename T>
struct type_caster<T, enable_if_t<std::is_arithmetic<T>::value && !is_std_char_type<T>::value>> {
    using _py_type_0 = conditional_t<sizeof(T) <= sizeof(long), long, long long>;
    using _py_type_1 = conditional_t<std::is_signed<T>::value, _py_type_0, typename std::make_unsigned<_py_type_0>::type>;
    using py_type = conditional_t<std::is_floating_point<T>::value, double, _py_type_1>;
public:

    bool load(handle src, bool convert) {
        py_type py_value;

        if (!src)
            return false;

        if (std::is_floating_point<T>::value) {
            if (convert || PyFloat_Check(src.ptr()))
                py_value = (py_type) PyFloat_AsDouble(src.ptr());
            else
                return false;
        } else if (PyFloat_Check(src.ptr())) {
            return false;
        } else if (std::is_unsigned<py_type>::value) {
            py_value = as_unsigned<py_type>(src.ptr());
        } else { // signed integer:
            py_value = sizeof(T) <= sizeof(long)
                ? (py_type) PyLong_AsLong(src.ptr())
                : (py_type) PYBIND11_LONG_AS_LONGLONG(src.ptr());
        }

        bool py_err = py_value == (py_type) -1 && PyErr_Occurred();
        if (py_err || (std::is_integral<T>::value && sizeof(py_type) != sizeof(T) &&
                       (py_value < (py_type) std::numeric_limits<T>::min() ||
                        py_value > (py_type) std::numeric_limits<T>::max()))) {
            bool type_error = py_err && PyErr_ExceptionMatches(
#if PY_VERSION_HEX < 0x03000000 && !defined(PYPY_VERSION)
                PyExc_SystemError
#else
                PyExc_TypeError
#endif
            );
            PyErr_Clear();
            if (type_error && convert && PyNumber_Check(src.ptr())) {
                auto tmp = reinterpret_steal<object>(std::is_floating_point<T>::value
                                                     ? PyNumber_Float(src.ptr())
                                                     : PyNumber_Long(src.ptr()));
                PyErr_Clear();
                return load(tmp, false);
            }
            return false;
        }

        value = (T) py_value;
        return true;
    }

    static handle cast(T src, return_value_policy /* policy */, handle /* parent */) {
        if (std::is_floating_point<T>::value) {
            return PyFloat_FromDouble((double) src);
        } else if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                return PyLong_FromLong((long) src);
            else
                return PyLong_FromUnsignedLong((unsigned long) src);
        } else {
            if (std::is_signed<T>::value)
                return PyLong_FromLongLong((long long) src);
            else
                return PyLong_FromUnsignedLongLong((unsigned long long) src);
        }
    }

    PYBIND11_TYPE_CASTER(T, _<std::is_integral<T>::value>("int", "float"));
};

template<typename T> struct void_caster {
public:
    bool load(handle src, bool) {
        if (src && src.is_none())
            return true;
        return false;
    }
    static handle cast(T, return_value_policy /* policy */, handle /* parent */) {
        return none().inc_ref();
    }
    PYBIND11_TYPE_CASTER(T, _("None"));
};

template <> class type_caster<void_type> : public void_caster<void_type> {};

template <> class type_caster<void> : public type_caster<void_type> {
public:
    using type_caster<void_type>::cast;

    bool load(handle h, bool) {
        if (!h) {
            return false;
        } else if (h.is_none()) {
            value = nullptr;
            return true;
        }

        /* Check if this is a capsule */
        if (isinstance<capsule>(h)) {
            value = reinterpret_borrow<capsule>(h);
            return true;
        }

        /* Check if this is a C++ type */
        auto &bases = all_type_info((PyTypeObject *) h.get_type().ptr());
        if (bases.size() == 1) { // Only allowing loading from a single-value type
            value = values_and_holders(reinterpret_cast<instance *>(h.ptr())).begin()->value_ptr();
            return true;
        }

        /* Fail */
        return false;
    }

    static handle cast(const void *ptr, return_value_policy /* policy */, handle /* parent */) {
        if (ptr)
            return capsule(ptr).release();
        else
            return none().inc_ref();
    }

    template <typename T> using cast_op_type = void*&;
    operator void *&() { return value; }
    static PYBIND11_DESCR name() { return type_descr(_("capsule")); }
private:
    void *value = nullptr;
};

template <> class type_caster<std::nullptr_t> : public void_caster<std::nullptr_t> { };

template <> class type_caster<bool> {
public:
    bool load(handle src, bool convert) {
        if (!src) return false;
        else if (src.ptr() == Py_True) { value = true; return true; }
        else if (src.ptr() == Py_False) { value = false; return true; }
        else if (convert || !strcmp("numpy.bool_", Py_TYPE(src.ptr())->tp_name)) {
            // (allow non-implicit conversion for numpy booleans)

            Py_ssize_t res = -1;
            if (src.is_none()) {
                res = 0;  // None is implicitly converted to False
            }
            #if defined(PYPY_VERSION)
            // On PyPy, check that "__bool__" (or "__nonzero__" on Python 2.7) attr exists
            else if (hasattr(src, PYBIND11_BOOL_ATTR)) {
                res = PyObject_IsTrue(src.ptr());
            }
            #else
            // Alternate approach for CPython: this does the same as the above, but optimized
            // using the CPython API so as to avoid an unneeded attribute lookup.
            else if (auto tp_as_number = src.ptr()->ob_type->tp_as_number) {
                if (PYBIND11_NB_BOOL(tp_as_number)) {
                    res = (*PYBIND11_NB_BOOL(tp_as_number))(src.ptr());
                }
            }
            #endif
            if (res == 0 || res == 1) {
                value = (bool) res;
                return true;
            }
        }
        return false;
    }
    static handle cast(bool src, return_value_policy /* policy */, handle /* parent */) {
        return handle(src ? Py_True : Py_False).inc_ref();
    }
    PYBIND11_TYPE_CASTER(bool, _("bool"));
};

// Helper class for UTF-{8,16,32} C++ stl strings:
template <typename StringType, bool IsView = false> struct string_caster {
    using CharT = typename StringType::value_type;

    // Simplify life by being able to assume standard char sizes (the standard only guarantees
    // minimums, but Python requires exact sizes)
    static_assert(!std::is_same<CharT, char>::value || sizeof(CharT) == 1, "Unsupported char size != 1");
    static_assert(!std::is_same<CharT, char16_t>::value || sizeof(CharT) == 2, "Unsupported char16_t size != 2");
    static_assert(!std::is_same<CharT, char32_t>::value || sizeof(CharT) == 4, "Unsupported char32_t size != 4");
    // wchar_t can be either 16 bits (Windows) or 32 (everywhere else)
    static_assert(!std::is_same<CharT, wchar_t>::value || sizeof(CharT) == 2 || sizeof(CharT) == 4,
            "Unsupported wchar_t size != 2/4");
    static constexpr size_t UTF_N = 8 * sizeof(CharT);

    bool load(handle src, bool) {
#if PY_MAJOR_VERSION < 3
        object temp;
#endif
        handle load_src = src;
        if (!src) {
            return false;
        } else if (!PyUnicode_Check(load_src.ptr())) {
#if PY_MAJOR_VERSION >= 3
            return load_bytes(load_src);
#else
            if (sizeof(CharT) == 1) {
                return load_bytes(load_src);
            }

            // The below is a guaranteed failure in Python 3 when PyUnicode_Check returns false
            if (!PYBIND11_BYTES_CHECK(load_src.ptr()))
                return false;

            temp = reinterpret_steal<object>(PyUnicode_FromObject(load_src.ptr()));
            if (!temp) { PyErr_Clear(); return false; }
            load_src = temp;
#endif
        }

        object utfNbytes = reinterpret_steal<object>(PyUnicode_AsEncodedString(
            load_src.ptr(), UTF_N == 8 ? "utf-8" : UTF_N == 16 ? "utf-16" : "utf-32", nullptr));
        if (!utfNbytes) { PyErr_Clear(); return false; }

        const CharT *buffer = reinterpret_cast<const CharT *>(PYBIND11_BYTES_AS_STRING(utfNbytes.ptr()));
        size_t length = (size_t) PYBIND11_BYTES_SIZE(utfNbytes.ptr()) / sizeof(CharT);
        if (UTF_N > 8) { buffer++; length--; } // Skip BOM for UTF-16/32
        value = StringType(buffer, length);

        // If we're loading a string_view we need to keep the encoded Python object alive:
        if (IsView)
            loader_life_support::add_patient(utfNbytes);

        return true;
    }

    static handle cast(const StringType &src, return_value_policy /* policy */, handle /* parent */) {
        const char *buffer = reinterpret_cast<const char *>(src.data());
        ssize_t nbytes = ssize_t(src.size() * sizeof(CharT));
        handle s = decode_utfN(buffer, nbytes);
        if (!s) throw error_already_set();
        return s;
    }

    PYBIND11_TYPE_CASTER(StringType, _(PYBIND11_STRING_NAME));

private:
    static handle decode_utfN(const char *buffer, ssize_t nbytes) {
#if !defined(PYPY_VERSION)
        return
            UTF_N == 8  ? PyUnicode_DecodeUTF8(buffer, nbytes, nullptr) :
            UTF_N == 16 ? PyUnicode_DecodeUTF16(buffer, nbytes, nullptr, nullptr) :
                          PyUnicode_DecodeUTF32(buffer, nbytes, nullptr, nullptr);
#else
        // PyPy seems to have multiple problems related to PyUnicode_UTF*: the UTF8 version
        // sometimes segfaults for unknown reasons, while the UTF16 and 32 versions require a
        // non-const char * arguments, which is also a nuissance, so bypass the whole thing by just
        // passing the encoding as a string value, which works properly:
        return PyUnicode_Decode(buffer, nbytes, UTF_N == 8 ? "utf-8" : UTF_N == 16 ? "utf-16" : "utf-32", nullptr);
#endif
    }

    // When loading into a std::string or char*, accept a bytes object as-is (i.e.
    // without any encoding/decoding attempt).  For other C++ char sizes this is a no-op.
    // which supports loading a unicode from a str, doesn't take this path.
    template <typename C = CharT>
    bool load_bytes(enable_if_t<sizeof(C) == 1, handle> src) {
        if (PYBIND11_BYTES_CHECK(src.ptr())) {
            // We were passed a Python 3 raw bytes; accept it into a std::string or char*
            // without any encoding attempt.
            const char *bytes = PYBIND11_BYTES_AS_STRING(src.ptr());
            if (bytes) {
                value = StringType(bytes, (size_t) PYBIND11_BYTES_SIZE(src.ptr()));
                return true;
            }
        }

        return false;
    }

    template <typename C = CharT>
    bool load_bytes(enable_if_t<sizeof(C) != 1, handle>) { return false; }
};

template <typename CharT, class Traits, class Allocator>
struct type_caster<std::basic_string<CharT, Traits, Allocator>, enable_if_t<is_std_char_type<CharT>::value>>
    : string_caster<std::basic_string<CharT, Traits, Allocator>> {};

#ifdef PYBIND11_HAS_STRING_VIEW
template <typename CharT, class Traits>
struct type_caster<std::basic_string_view<CharT, Traits>, enable_if_t<is_std_char_type<CharT>::value>>
    : string_caster<std::basic_string_view<CharT, Traits>, true> {};
#endif

// Type caster for C-style strings.  We basically use a std::string type caster, but also add the
// ability to use None as a nullptr char* (which the string caster doesn't allow).
template <typename CharT> struct type_caster<CharT, enable_if_t<is_std_char_type<CharT>::value>> {
    using StringType = std::basic_string<CharT>;
    using StringCaster = type_caster<StringType>;
    StringCaster str_caster;
    bool none = false;
public:
    bool load(handle src, bool convert) {
        if (!src) return false;
        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) return false;
            none = true;
            return true;
        }
        return str_caster.load(src, convert);
    }

    static handle cast(const CharT *src, return_value_policy policy, handle parent) {
        if (src == nullptr) return pybind11::none().inc_ref();
        return StringCaster::cast(StringType(src), policy, parent);
    }

    static handle cast(CharT src, return_value_policy policy, handle parent) {
        if (std::is_same<char, CharT>::value) {
            handle s = PyUnicode_DecodeLatin1((const char *) &src, 1, nullptr);
            if (!s) throw error_already_set();
            return s;
        }
        return StringCaster::cast(StringType(1, src), policy, parent);
    }

    operator CharT*() { return none ? nullptr : const_cast<CharT *>(static_cast<StringType &>(str_caster).c_str()); }
    operator CharT() {
        if (none)
            throw value_error("Cannot convert None to a character");

        auto &value = static_cast<StringType &>(str_caster);
        size_t str_len = value.size();
        if (str_len == 0)
            throw value_error("Cannot convert empty string to a character");

        // If we're in UTF-8 mode, we have two possible failures: one for a unicode character that
        // is too high, and one for multiple unicode characters (caught later), so we need to figure
        // out how long the first encoded character is in bytes to distinguish between these two
        // errors.  We also allow want to allow unicode characters U+0080 through U+00FF, as those
        // can fit into a single char value.
        if (StringCaster::UTF_N == 8 && str_len > 1 && str_len <= 4) {
            unsigned char v0 = static_cast<unsigned char>(value[0]);
            size_t char0_bytes = !(v0 & 0x80) ? 1 : // low bits only: 0-127
                (v0 & 0xE0) == 0xC0 ? 2 : // 0b110xxxxx - start of 2-byte sequence
                (v0 & 0xF0) == 0xE0 ? 3 : // 0b1110xxxx - start of 3-byte sequence
                4; // 0b11110xxx - start of 4-byte sequence

            if (char0_bytes == str_len) {
                // If we have a 128-255 value, we can decode it into a single char:
                if (char0_bytes == 2 && (v0 & 0xFC) == 0xC0) { // 0x110000xx 0x10xxxxxx
                    return static_cast<CharT>(((v0 & 3) << 6) + (static_cast<unsigned char>(value[1]) & 0x3F));
                }
                // Otherwise we have a single character, but it's > U+00FF
                throw value_error("Character code point not in range(0x100)");
            }
        }

        // UTF-16 is much easier: we can only have a surrogate pair for values above U+FFFF, thus a
        // surrogate pair with total length 2 instantly indicates a range error (but not a "your
        // string was too long" error).
        else if (StringCaster::UTF_N == 16 && str_len == 2) {
            char16_t v0 = static_cast<char16_t>(value[0]);
            if (v0 >= 0xD800 && v0 < 0xE000)
                throw value_error("Character code point not in range(0x10000)");
        }

        if (str_len != 1)
            throw value_error("Expected a character, but multi-character string found");

        return value[0];
    }

    static PYBIND11_DESCR name() { return type_descr(_(PYBIND11_STRING_NAME)); }
    template <typename _T> using cast_op_type = remove_reference_t<pybind11::detail::cast_op_type<_T>>;
};

// Base implementation for std::tuple and std::pair
template <template<typename...> class Tuple, typename... Ts> class tuple_caster {
    using type = Tuple<Ts...>;
    static constexpr auto size = sizeof...(Ts);
    using indices = make_index_sequence<size>;
public:

    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src))
            return false;
        const auto seq = reinterpret_borrow<sequence>(src);
        if (seq.size() != size)
            return false;
        return load_impl(seq, convert, indices{});
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        return cast_impl(std::forward<T>(src), policy, parent, indices{});
    }

    static PYBIND11_DESCR name() {
        return type_descr(_("Tuple[") + detail::concat(make_caster<Ts>::name()...) + _("]"));
    }

    template <typename T> using cast_op_type = type;

    operator type() & { return implicit_cast(indices{}); }
    operator type() && { return std::move(*this).implicit_cast(indices{}); }

protected:
    template <size_t... Is>
    type implicit_cast(index_sequence<Is...>) & { return type(cast_op<Ts>(std::get<Is>(subcasters))...); }
    template <size_t... Is>
    type implicit_cast(index_sequence<Is...>) && { return type(cast_op<Ts>(std::move(std::get<Is>(subcasters)))...); }

    static constexpr bool load_impl(const sequence &, bool, index_sequence<>) { return true; }

    template <size_t... Is>
    bool load_impl(const sequence &seq, bool convert, index_sequence<Is...>) {
        for (bool r : {std::get<Is>(subcasters).load(seq[Is], convert)...})
            if (!r)
                return false;
        return true;
    }

    /* Implementation: Convert a C++ tuple into a Python tuple */
    template <typename T, size_t... Is>
    static handle cast_impl(T &&src, return_value_policy policy, handle parent, index_sequence<Is...>) {
        std::array<object, size> entries{{
            reinterpret_steal<object>(make_caster<Ts>::cast(std::get<Is>(std::forward<T>(src)), policy, parent))...
        }};
        for (const auto &entry: entries)
            if (!entry)
                return handle();
        tuple result(size);
        int counter = 0;
        for (auto & entry: entries)
            PyTuple_SET_ITEM(result.ptr(), counter++, entry.release().ptr());
        return result.release();
    }

    Tuple<make_caster<Ts>...> subcasters;
};

template <typename T1, typename T2> class type_caster<std::pair<T1, T2>>
    : public tuple_caster<std::pair, T1, T2> {};

template <typename... Ts> class type_caster<std::tuple<Ts...>>
    : public tuple_caster<std::tuple, Ts...> {};

/// Helper class which abstracts away certain actions. Users can provide specializations for
/// custom holders, but it's only necessary if the type has a non-standard interface.
template <typename T>
struct holder_helper {
    static auto get(const T &p) -> decltype(p.get()) { return p.get(); }
};

inline const detail::type_info* get_lowest_type(handle src, bool do_throw = true) {
    auto* py_type = (PyTypeObject*)src.get_type().ptr();
    return detail::get_type_info(py_type, do_throw);
}


/// Type caster for holder types like std::shared_ptr, etc.
template <typename type, typename holder_type>
struct copyable_holder_caster : public type_caster_base<type> {
public:
    using base = type_caster_base<type>;
    static_assert(std::is_base_of<base, type_caster<type>>::value,
            "Holder classes are only supported for custom types");
    using base::base;
    using base::cast;
    using base::typeinfo;
    using base::value;

    handle src;

    bool load(handle src_in, bool convert) {
        src = src_in;
        return base::template load_impl<copyable_holder_caster<type, holder_type>>(src, convert);
    }

    explicit operator type*() { return this->value; }
    explicit operator type&() { return *(this->value); }
    explicit operator holder_type*() { return &holder; }

    // Workaround for Intel compiler bug
    // see pybind11 issue 94
    #if defined(__ICC) || defined(__INTEL_COMPILER)
    operator holder_type&() { return holder; }
    #else
    explicit operator holder_type&() { return holder; }
    #endif

    // Risk increasing the `shared_ptr` ref count temporarily to maintain writeable
    // semantics without too much `const_cast<>` ooginess.
    static handle cast(holder_type &&src, return_value_policy, handle) {
        const auto *ptr = holder_helper<holder_type>::get(src);
        return type_caster_base<type>::cast_holder(ptr, holder_erased(&src));
    }

    static handle cast(const holder_type &src, return_value_policy, handle) {
        const auto *ptr = holder_helper<holder_type>::get(src);
        return type_caster_base<type>::cast_holder(ptr, holder_erased(&src));
    }

  // TODO(eric.cousineau): Define cast_op_type???

protected:
    friend class type_caster_generic;
    void check_holder_compat() {
        if (typeinfo->default_holder)
            throw cast_error("Unable to load a custom holder type from a default-holder instance");
    }

    bool load_value(value_and_holder &&v_h, LoadType load_type) {
        holder_type& v_holder = v_h.holder<holder_type>();
        bool do_release_to_cpp = false;
        const type_info* lowest_type = nullptr;
        if (src.ref_count() == 1 && load_type == LoadType::DerivedCppSinglePySingle) {
            // Go ahead and release ownership to C++, if able.
            auto* py_type = (PyTypeObject*)src.get_type().ptr();
            lowest_type = detail::get_type_info(py_type);
            // Double-check that we did not get along C++ inheritance.
            // TODO(eric.cousineau): Generalize this to unique_ptr<> case as well.
            const bool is_actually_pure_cpp = lowest_type->type == py_type;
            if (!is_actually_pure_cpp) {
                if (lowest_type->release_info.can_derive_from_wrapper) {
                    do_release_to_cpp = true;
                } else {
                    std::cerr << "WARNING! Casting to std::shared_ptr<> will cause Python subclass of pybind11 C++ instance to lose its Python portion. "
                                 "Make your base class extend from pybind11::wrapper<> to prevent aliasing."
                              << std::endl;
                }
            }
        }

        if (v_h.holder_constructed()) {
            value = v_h.value_ptr();
            // Don't need to worry about double-counting the shared_ptr stuff.
            holder = v_holder;
        } else {
            throw cast_error("Unable to cast from non-held to held instance (T& to Holder<T>) "
#if defined(NDEBUG)
                             "(compile in debug mode for type information)");
#else
                             "of type '" + type_id<holder_type>() + "''");
#endif
        }

        // Release *after* we already have
        if (do_release_to_cpp) {
            assert(v_h.inst->owned);
            assert(lowest_type->release_info.release_to_cpp);
            // Increase reference count to pass to release mechanism.
            object obj = reinterpret_borrow<object>(src);
            lowest_type->release_info.release_to_cpp(v_h.inst, &holder, std::move(obj));
        }

        return true;
    }

    template <typename T = holder_type, detail::enable_if_t<!std::is_constructible<T, const T &, type*>::value, int> = 0>
    bool try_implicit_casts(handle, bool) { return false; }

    template <typename T = holder_type, detail::enable_if_t<std::is_constructible<T, const T &, type*>::value, int> = 0>
    bool try_implicit_casts(handle src, bool convert) {
        for (auto &cast : typeinfo->implicit_casts) {
            copyable_holder_caster sub_caster(*cast.first);
            if (sub_caster.load(src, convert)) {
                value = cast.second(sub_caster.value);
                holder = holder_type(sub_caster.holder, (type *) value);
                return true;
            }
        }
        return false;
    }

    static bool try_direct_conversions(handle) { return false; }


    holder_type holder;
    constexpr static detail::HolderTypeId holder_type_id = detail::get_holder_type_id<holder_type>::value;
};

/// Specialize for the common std::shared_ptr, so users don't need to
template <typename T>
class type_caster<std::shared_ptr<T>> : public copyable_holder_caster<T, std::shared_ptr<T>> { };

template <typename type, typename holder_type>
struct move_only_holder_caster : type_caster_base<type> {
        using base = type_caster_base<type>;
    static_assert(std::is_base_of<base, type_caster<type>>::value,
            "Holder classes are only supported for custom types");
    using base::base;
    using base::cast;
    using base::typeinfo;
    using base::value;

    static_assert(std::is_base_of<type_caster_base<type>, type_caster<type>>::value,
            "Holder classes are only supported for custom types");

    static handle cast(holder_type &&src, return_value_policy, handle) {
        // Move `src` so that `holder_helper<>::get()` can call `release` if need be.
        // That way, if we mix `holder_type`s, we don't have to worry about `existing_holder`
        // from being mistakenly reinterpret_cast'd to `shared_ptr<type>` (#1138).
        auto *ptr = holder_helper<holder_type>::get(src);
        return type_caster_base<type>::cast_holder(ptr, holder_erased(&src));
    }

  // Disable these?
//  explicit operator type*() { return this->value; }
//  explicit operator type&() { return *(this->value); }

//  explicit operator holder_type*() { return &holder; }

  // Force rvalue.
  template <typename T>
  using cast_op_type = holder_type&&;

  explicit operator holder_type&&() { return std::move(holder); }

    object extract_from_container(handle src) {
        // See if this is a supported `move` container.
        if (isinstance(src, (PyObject*)&PyList_Type) && PyList_Size(src.ptr()) == 1) {
            // Extract the object from a single-item list, and remove the existing reference so we have exclusive control.
            // @note This will break implicit casting when constructing from vectors, but eh, who cares.
            // Swap.
            list li = src.cast<list>();
            object obj = li[0];
            li[0] = none();
            return obj;
        } else if (hasattr(src, "_is_move_container")) {
            // Try to extract the value with `release()`.
            return src.attr("release")();
        } else {
            throw std::runtime_error(
                "Only use cast<unique_ptr<T>>() with a Python move-container (such as a single-item list), "
                "or ensure that you call cast<unique_ptr<T>(std::move(obj))");
        }
    }

    bool load(handle src, bool convert) {
        // Ensure that we have exclusive control (with `object` reference count control) over the entering object.
        // That way, we maintain complete control, and do not need to worry about stacked function calls.

        // TODO(eric.cousineau): Inconsistency will be absolutely painful. Thinking about, move'ing the return value of a function
        // may work with the exact type desired, but it most likely *won't* work if the type is wrapped in a `move` container.
        // Solution: Check more aggressively if this is a move container.

        object obj_exclusive = extract_from_container(src);
        if (obj_exclusive.ref_count() != 1) {
            throw std::runtime_error("Non-unique reference, cannot cast to unique_ptr.");
        }

        // TODO(eric.cousineau): To reduce number of references, require that a list be passed in,
        // such that the value can be set to zero.
        // That, or figure out where the `py::object` is in this call chain, and set that to zero.
        // Specialize for unique_ptr<>?

        // Do not use `load_impl`, as it's not structured conveniently for `unique_ptr`.
        // Specifically, trying to delegate to resolving to conversion.
        // return base::template load_impl<move_only_holder_caster<type, holder_type>>(src, convert);
        check_holder_compat();
        auto v_h = reinterpret_cast<instance *>(obj_exclusive.ptr())->get_value_and_holder();
        LoadType load_type = determine_load_type(obj_exclusive, typeinfo);
        return load_value(std::move(obj_exclusive), std::move(v_h), load_type);
    }

    static PYBIND11_DESCR name() { return type_caster_base<type>::name(); }

protected:
    friend class type_caster_generic;
    void check_holder_compat() {
        if (!typeinfo->default_holder)
            throw cast_error("Unable to load a non-default holder type (unique_ptr)");
    }

    bool load_value(object obj_exclusive, value_and_holder &&v_h, LoadType load_type) {
        // TODO(eric.cousineau): This should try and find the downcast-lowest
        // level (closest to child) `release_to_cpp` method that is derived-releasable
        // (which derives from `wrapper<type>`).
        // This should resolve general casting, and should also avoid alias
        // branch issues:
        //   Example: `Base` has wrapper `PyBase` which extends `wrapper<Base>`.
        //   `Child` extends `Base`, has its own wrapper `PyChild`, which extends
        //   `wrapper<Child>`.
        //   Anything deriving from `Child` does not derive from `PyBase`, so we should
        //   NOT try to release using `PyBase`s mechanism.
        //   Additionally, if `Child` does not have a wrapper (for whatever reason) and is extended,
        //   then we still can NOT use `PyBase` since it's not part of the hierachy.

        // Try to get the lowest-hierarchy level of the type.
        // This requires that we are single-inheritance at most.
        const detail::type_info* lowest_type = nullptr;
        switch (load_type) {
            case LoadType::PureCpp: {
                // We already have the lowest type.
                lowest_type = typeinfo;
                break;
            }
            // If the base type is explicitly mentioned, then we can rely on `DerivedCppSinglePySingle` being used.
            case LoadType::DerivedCppSinglePySingle:
                // However, if it is not, it may be that we have a C++ type inheriting from another C++ type without the inheritance being registered.
                // In this case, we delegate by effectively downcasting in Python by finding the lowest-level type.
                // @note No `break` here on purpose!
            case LoadType::ConversionNeeded: {
                    // Try to get the lowest-hierarchy (closets to child class) of the type.
                // The usage of `get_type_info` implicitly requires single inheritance.
                auto* py_type = (PyTypeObject*)obj_exclusive.get_type().ptr();
                lowest_type = detail::get_type_info(py_type);
                break;
            }
            default: {
                throw std::runtime_error("Unsupported load type");
            }
        }
        if (!lowest_type)
            throw std::runtime_error("No valid lowest type. Internal error?");
        auto& release_info = lowest_type->release_info;
        if (!release_info.release_to_cpp)
            throw std::runtime_error("No release mechanism in lowest type?");
        release_info.release_to_cpp(v_h.inst, &holder, std::move(obj_exclusive));
        return true;
    }

    holder_type holder;
};

template <typename type, typename deleter>
class type_caster<std::unique_ptr<type, deleter>>
    : public move_only_holder_caster<type, std::unique_ptr<type, deleter>> { };

template <typename type, typename holder_type>
using type_caster_holder = conditional_t<is_copy_constructible<holder_type>::value,
                                         copyable_holder_caster<type, holder_type>,
                                         move_only_holder_caster<type, holder_type>>;

template <typename T, bool Value = false> struct always_construct_holder { static constexpr bool value = Value; };

/// Create a specialization for custom holder types (silently ignores std::shared_ptr)
#define PYBIND11_DECLARE_HOLDER_TYPE(type, holder_type, ...) \
    namespace pybind11 { namespace detail { \
    template <typename type> \
    struct always_construct_holder<holder_type> : always_construct_holder<void, ##__VA_ARGS__>  { }; \
    template <typename type> \
    class type_caster<holder_type, enable_if_t<!is_shared_ptr<holder_type>::value>> \
        : public type_caster_holder<type, holder_type> { }; \
    }}

// PYBIND11_DECLARE_HOLDER_TYPE holder types:
template <typename base, typename holder> struct is_holder_type :
    std::is_base_of<detail::type_caster_holder<base, holder>, detail::type_caster<holder>> {};
// Specialization for always-supported unique_ptr holders:
template <typename base, typename deleter> struct is_holder_type<base, std::unique_ptr<base, deleter>> :
    std::true_type {};

template <typename T> struct handle_type_name { static PYBIND11_DESCR name() { return _<T>(); } };
template <> struct handle_type_name<bytes> { static PYBIND11_DESCR name() { return _(PYBIND11_BYTES_NAME); } };
template <> struct handle_type_name<args> { static PYBIND11_DESCR name() { return _("*args"); } };
template <> struct handle_type_name<kwargs> { static PYBIND11_DESCR name() { return _("**kwargs"); } };

template <typename type>
struct pyobject_caster {
    template <typename T = type, enable_if_t<std::is_same<T, handle>::value, int> = 0>
    bool load(handle src, bool /* convert */) { value = src; return static_cast<bool>(value); }

    template <typename T = type, enable_if_t<std::is_base_of<object, T>::value, int> = 0>
    bool load(handle src, bool /* convert */) {
        if (!isinstance<type>(src))
            return false;
        value = reinterpret_borrow<type>(src);
        return true;
    }

    static handle cast(const handle &src, return_value_policy /* policy */, handle /* parent */) {
        return src.inc_ref();
    }
    PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
};

template <typename T>
class type_caster<T, enable_if_t<is_pyobject<T>::value>> : public pyobject_caster<T> { };

// Our conditions for enabling moving are quite restrictive:
// At compile time:
// - T needs to be a non-const, non-pointer, non-reference type
// - type_caster<T>::operator T&() must exist
// - the type must be move constructible (obviously)
// At run-time:
// - if the type is non-copy-constructible, the object must be the sole owner of the type (i.e. it
//   must have ref_count() == 1)h
// If any of the above are not satisfied, we fall back to copying.
template <typename T> using move_is_plain_type = satisfies_none_of<T,
    std::is_void, std::is_pointer, std::is_reference, std::is_const
>;
template <typename T, typename SFINAE = void> struct move_always : std::false_type {};
template <typename T> struct move_always<T, enable_if_t<all_of<
    move_is_plain_type<T>,
    negation<is_copy_constructible<T>>,
    std::is_move_constructible<T>,
    std::is_same<decltype(std::declval<make_caster<T>>().operator T&()), T&>
>::value>> : std::true_type {};
template <typename T, typename SFINAE = void> struct move_if_unreferenced : std::false_type {};
template <typename T> struct move_if_unreferenced<T, enable_if_t<all_of<
    move_is_plain_type<T>,
    negation<move_always<T>>,
    std::is_move_constructible<T>,
    std::is_same<decltype(std::declval<make_caster<T>>().operator T&()), T&>
>::value>> : std::true_type {};
template <typename T> using move_never = none_of<move_always<T>, move_if_unreferenced<T>>;

// Detect whether returning a `type` from a cast on type's type_caster is going to result in a
// reference or pointer to a local variable of the type_caster.  Basically, only
// non-reference/pointer `type`s and reference/pointers from a type_caster_generic are safe;
// everything else returns a reference/pointer to a local variable.
template <typename type> using cast_is_temporary_value_reference = bool_constant<
    (std::is_reference<type>::value || std::is_pointer<type>::value) &&
    !std::is_base_of<type_caster_generic, make_caster<type>>::value
>;

// When a value returned from a C++ function is being cast back to Python, we almost always want to
// force `policy = move`, regardless of the return value policy the function/method was declared
// with.  Some classes (most notably Eigen::Ref and related) need to avoid this, and so can do so by
// specializing this struct.
template <typename Return, typename SFINAE = void> struct return_value_policy_override {
    static return_value_policy policy(return_value_policy p) {
        return !std::is_lvalue_reference<Return>::value && !std::is_pointer<Return>::value
            ? return_value_policy::move : p;
    }
};

// Basic python -> C++ casting; throws if casting fails
template <typename T, typename SFINAE> type_caster<T, SFINAE> &load_type(type_caster<T, SFINAE> &conv, const handle &handle) {
    if (!conv.load(handle, true)) {
#if defined(NDEBUG)
        throw cast_error("Unable to cast Python instance to C++ type (compile in debug mode for details)");
#else
        throw cast_error("Unable to cast Python instance of type " +
            (std::string) str(handle.get_type()) + " to C++ type '" + type_id<T>() + "''");
#endif
    }
    return conv;
}
// Wrapper around the above that also constructs and returns a type_caster
template <typename T> make_caster<T> load_type(const handle &handle) {
    make_caster<T> conv;
    load_type(conv, handle);
    return conv;
}

NAMESPACE_END(detail)

// pytype -> C++ type
template <typename T, detail::enable_if_t<!detail::is_pyobject<T>::value, int> = 0>
T cast(const handle &handle) {
    using namespace detail;
    static_assert(!cast_is_temporary_value_reference<T>::value,
            "Unable to cast type to reference: value is local to type caster");
    return cast_op<T>(load_type<T>(handle));
}

// pytype -> pytype (calls converting constructor)
template <typename T, detail::enable_if_t<detail::is_pyobject<T>::value, int> = 0>
T cast(const handle &handle) { return T(reinterpret_borrow<object>(handle)); }

// C++ type -> py::object
template <typename T, detail::enable_if_t<!detail::is_pyobject<T>::value, int> = 0>
object cast(const T &value, return_value_policy policy = return_value_policy::automatic_reference,
            handle parent = handle()) {
    if (policy == return_value_policy::automatic)
        policy = std::is_pointer<T>::value ? return_value_policy::take_ownership : return_value_policy::copy;
    else if (policy == return_value_policy::automatic_reference)
        policy = std::is_pointer<T>::value ? return_value_policy::reference : return_value_policy::copy;
    return reinterpret_steal<object>(detail::make_caster<T>::cast(value, policy, parent));
}

template <typename T> T handle::cast() const { return pybind11::cast<T>(*this); }
template <> inline void handle::cast() const { return; }

template <typename T>
detail::enable_if_t<!detail::move_never<T>::value, T> move(object &&obj) {
    if (obj.ref_count() > 1)
#if defined(NDEBUG)
        throw cast_error("Unable to cast Python instance to C++ rvalue: instance has multiple references"
            " (compile in debug mode for details)");
#else
        throw cast_error("Unable to move from Python " + (std::string) str(obj.get_type()) +
                " instance to C++ " + type_id<T>() + " instance: instance has multiple references");
#endif

    // Move into a temporary and return that, because the reference may be a local value of `conv`
    T ret = std::move(detail::load_type<T>(obj).operator T&());
    return ret;
}

// Calling cast() on an rvalue calls pybind::cast with the object rvalue, which does:
// - If we have to move (because T has no copy constructor), do it.  This will fail if the moved
//   object has multiple references, but trying to copy will fail to compile.
// - If both movable and copyable, check ref count: if 1, move; otherwise copy
// - Otherwise (not movable), copy.
template <typename T> detail::enable_if_t<detail::move_always<T>::value, T> cast(object &&object) {
    return move<T>(std::move(object));
}
template <typename T> detail::enable_if_t<detail::move_if_unreferenced<T>::value, T> cast(object &&object) {
    if (object.ref_count() > 1)
        return cast<T>(object);
    else
        return move<T>(std::move(object));
}
template <typename T> detail::enable_if_t<detail::move_never<T>::value, T> cast(object &&object) {
    return cast<T>(object);
}

template <typename T> T object::cast() const & { return pybind11::cast<T>(*this); }
template <typename T> T object::cast() && { return pybind11::cast<T>(std::move(*this)); }
template <> inline void object::cast() const & { return; }
template <> inline void object::cast() && { return; }

NAMESPACE_BEGIN(detail)

// Declared in pytypes.h:
template <typename T, enable_if_t<!is_pyobject<T>::value, int>>
object object_or_cast(T &&o) { return pybind11::cast(std::forward<T>(o)); }

struct overload_unused {}; // Placeholder type for the unneeded (and dead code) static variable in the OVERLOAD_INT macro
template <typename ret_type> using overload_caster_t = conditional_t<
    cast_is_temporary_value_reference<ret_type>::value, make_caster<ret_type>, overload_unused>;

// Trampoline use: for reference/pointer types to value-converted values, we do a value cast, then
// store the result in the given variable.  For other types, this is a no-op.
template <typename T> enable_if_t<cast_is_temporary_value_reference<T>::value, T> cast_ref(object &&o, make_caster<T> &caster) {
    return cast_op<T>(load_type(caster, o));
}
template <typename T> enable_if_t<!cast_is_temporary_value_reference<T>::value, T> cast_ref(object &&, overload_unused &) {
    pybind11_fail("Internal error: cast_ref fallback invoked"); }

// Trampoline use: Having a pybind11::cast with an invalid reference type is going to static_assert, even
// though if it's in dead code, so we provide a "trampoline" to pybind11::cast that only does anything in
// cases where pybind11::cast is valid.
template <typename T> enable_if_t<!cast_is_temporary_value_reference<T>::value, T> cast_safe(object &&o) {
    return pybind11::cast<T>(std::move(o)); }
template <typename T> enable_if_t<cast_is_temporary_value_reference<T>::value, T> cast_safe(object &&) {
    pybind11_fail("Internal error: cast_safe fallback invoked"); }
template <> inline void cast_safe<void>(object &&) {}

NAMESPACE_END(detail)

template <return_value_policy policy = return_value_policy::automatic_reference,
          typename... Args> tuple make_tuple(Args&&... args_) {
    constexpr size_t size = sizeof...(Args);
    std::array<object, size> args {
        { reinterpret_steal<object>(detail::make_caster<Args>::cast(
            std::forward<Args>(args_), policy, nullptr))... }
    };
    for (size_t i = 0; i < args.size(); i++) {
        if (!args[i]) {
#if defined(NDEBUG)
            throw cast_error("make_tuple(): unable to convert arguments to Python object (compile in debug mode for details)");
#else
            std::array<std::string, size> argtypes { {type_id<Args>()...} };
            throw cast_error("make_tuple(): unable to convert argument of type '" +
                argtypes[i] + "' to Python object");
#endif
        }
    }
    tuple result(size);
    int counter = 0;
    for (auto &arg_value : args)
        PyTuple_SET_ITEM(result.ptr(), counter++, arg_value.release().ptr());
    return result;
}

/// \ingroup annotations
/// Annotation for arguments
struct arg {
    /// Constructs an argument with the name of the argument; if null or omitted, this is a positional argument.
    constexpr explicit arg(const char *name = nullptr) : name(name), flag_noconvert(false), flag_none(true) { }
    /// Assign a value to this argument
    template <typename T> arg_v operator=(T &&value) const;
    /// Indicate that the type should not be converted in the type caster
    arg &noconvert(bool flag = true) { flag_noconvert = flag; return *this; }
    /// Indicates that the argument should/shouldn't allow None (e.g. for nullable pointer args)
    arg &none(bool flag = true) { flag_none = flag; return *this; }

    const char *name; ///< If non-null, this is a named kwargs argument
    bool flag_noconvert : 1; ///< If set, do not allow conversion (requires a supporting type caster!)
    bool flag_none : 1; ///< If set (the default), allow None to be passed to this argument
};

/// \ingroup annotations
/// Annotation for arguments with values
struct arg_v : arg {
private:
    template <typename T>
    arg_v(arg &&base, T &&x, const char *descr = nullptr)
        : arg(base),
          value(reinterpret_steal<object>(
              detail::make_caster<T>::cast(x, return_value_policy::automatic, {})
          )),
          descr(descr)
#if !defined(NDEBUG)
        , type(type_id<T>())
#endif
    { }

public:
    /// Direct construction with name, default, and description
    template <typename T>
    arg_v(const char *name, T &&x, const char *descr = nullptr)
        : arg_v(arg(name), std::forward<T>(x), descr) { }

    /// Called internally when invoking `py::arg("a") = value`
    template <typename T>
    arg_v(const arg &base, T &&x, const char *descr = nullptr)
        : arg_v(arg(base), std::forward<T>(x), descr) { }

    /// Same as `arg::noconvert()`, but returns *this as arg_v&, not arg&
    arg_v &noconvert(bool flag = true) { arg::noconvert(flag); return *this; }

    /// Same as `arg::nonone()`, but returns *this as arg_v&, not arg&
    arg_v &none(bool flag = true) { arg::none(flag); return *this; }

    /// The default value
    object value;
    /// The (optional) description of the default value
    const char *descr;
#if !defined(NDEBUG)
    /// The C++ type name of the default value (only available when compiled in debug mode)
    std::string type;
#endif
};

template <typename T>
arg_v arg::operator=(T &&value) const { return {std::move(*this), std::forward<T>(value)}; }

/// Alias for backward compatibility -- to be removed in version 2.0
template <typename /*unused*/> using arg_t = arg_v;

inline namespace literals {
/** \rst
    String literal version of `arg`
 \endrst */
constexpr arg operator"" _a(const char *name, size_t) { return arg(name); }
}

NAMESPACE_BEGIN(detail)

// forward declaration (definition in attr.h)
struct function_record;

/// Internal data associated with a single function call
struct function_call {
    function_call(function_record &f, handle p); // Implementation in attr.h

    /// The function data:
    const function_record &func;

    /// Arguments passed to the function:
    std::vector<handle> args;

    /// The `convert` value the arguments should be loaded with
    std::vector<bool> args_convert;

    /// The parent, if any
    handle parent;

    /// If this is a call to an initializer, this argument contains `self`
    handle init_self;
};


/// Helper class which loads arguments for C++ functions called from Python
template <typename... Args>
class argument_loader {
    using indices = make_index_sequence<sizeof...(Args)>;

    template <typename Arg> using argument_is_args   = std::is_same<intrinsic_t<Arg>, args>;
    template <typename Arg> using argument_is_kwargs = std::is_same<intrinsic_t<Arg>, kwargs>;
    // Get args/kwargs argument positions relative to the end of the argument list:
    static constexpr auto args_pos = constexpr_first<argument_is_args, Args...>() - (int) sizeof...(Args),
                        kwargs_pos = constexpr_first<argument_is_kwargs, Args...>() - (int) sizeof...(Args);

    static constexpr bool args_kwargs_are_last = kwargs_pos >= - 1 && args_pos >= kwargs_pos - 1;

    static_assert(args_kwargs_are_last, "py::args/py::kwargs are only permitted as the last argument(s) of a function");

public:
    static constexpr bool has_kwargs = kwargs_pos < 0;
    static constexpr bool has_args = args_pos < 0;

    static PYBIND11_DESCR arg_names() { return detail::concat(make_caster<Args>::name()...); }

    bool load_args(function_call &call) {
        return load_impl_sequence(call, indices{});
    }

    template <typename Return, typename Guard, typename Func>
    enable_if_t<!std::is_void<Return>::value, Return> call(Func &&f) && {
        return std::move(*this).template call_impl<Return>(std::forward<Func>(f), indices{}, Guard{});
    }

    template <typename Return, typename Guard, typename Func>
    enable_if_t<std::is_void<Return>::value, void_type> call(Func &&f) && {
        std::move(*this).template call_impl<Return>(std::forward<Func>(f), indices{}, Guard{});
        return void_type();
    }

private:

    static bool load_impl_sequence(function_call &, index_sequence<>) { return true; }

    template <size_t... Is>
    bool load_impl_sequence(function_call &call, index_sequence<Is...>) {
        for (bool r : {std::get<Is>(argcasters).load(call.args[Is], call.args_convert[Is])...})
            if (!r)
                return false;
        return true;
    }

    template <typename Return, typename Func, size_t... Is, typename Guard>
    Return call_impl(Func &&f, index_sequence<Is...>, Guard &&) {
        return std::forward<Func>(f)(cast_op<Args>(std::move(std::get<Is>(argcasters)))...);
    }

    std::tuple<make_caster<Args>...> argcasters;
};

/// Helper class which collects only positional arguments for a Python function call.
/// A fancier version below can collect any argument, but this one is optimal for simple calls.
template <return_value_policy policy>
class simple_collector {
public:
    template <typename... Ts>
    explicit simple_collector(Ts &&...values)
        : m_args(pybind11::make_tuple<policy>(std::forward<Ts>(values)...)) { }

    const tuple &args() const & { return m_args; }
    dict kwargs() const { return {}; }

    tuple args() && { return std::move(m_args); }

    /// Call a Python function and pass the collected arguments
    object call(PyObject *ptr) const {
        PyObject *result = PyObject_CallObject(ptr, m_args.ptr());
        if (!result)
            throw error_already_set();
        return reinterpret_steal<object>(result);
    }

private:
    tuple m_args;
};

/// Helper class which collects positional, keyword, * and ** arguments for a Python function call
template <return_value_policy policy>
class unpacking_collector {
public:
    template <typename... Ts>
    explicit unpacking_collector(Ts &&...values) {
        // Tuples aren't (easily) resizable so a list is needed for collection,
        // but the actual function call strictly requires a tuple.
        auto args_list = list();
        int _[] = { 0, (process(args_list, std::forward<Ts>(values)), 0)... };
        ignore_unused(_);

        m_args = std::move(args_list);
    }

    const tuple &args() const & { return m_args; }
    const dict &kwargs() const & { return m_kwargs; }

    tuple args() && { return std::move(m_args); }
    dict kwargs() && { return std::move(m_kwargs); }

    /// Call a Python function and pass the collected arguments
    object call(PyObject *ptr) const {
        PyObject *result = PyObject_Call(ptr, m_args.ptr(), m_kwargs.ptr());
        if (!result)
            throw error_already_set();
        return reinterpret_steal<object>(result);
    }

private:
    template <typename T>
    void process(list &args_list, T &&x) {
        auto o = reinterpret_steal<object>(detail::make_caster<T>::cast(std::forward<T>(x), policy, {}));
        if (!o) {
#if defined(NDEBUG)
            argument_cast_error();
#else
            argument_cast_error(std::to_string(args_list.size()), type_id<T>());
#endif
        }
        args_list.append(o);
    }

    void process(list &args_list, detail::args_proxy ap) {
        for (const auto &a : ap)
            args_list.append(a);
    }

    void process(list &/*args_list*/, arg_v a) {
        if (!a.name)
#if defined(NDEBUG)
            nameless_argument_error();
#else
            nameless_argument_error(a.type);
#endif

        if (m_kwargs.contains(a.name)) {
#if defined(NDEBUG)
            multiple_values_error();
#else
            multiple_values_error(a.name);
#endif
        }
        if (!a.value) {
#if defined(NDEBUG)
            argument_cast_error();
#else
            argument_cast_error(a.name, a.type);
#endif
        }
        m_kwargs[a.name] = a.value;
    }

    void process(list &/*args_list*/, detail::kwargs_proxy kp) {
        if (!kp)
            return;
        for (const auto &k : reinterpret_borrow<dict>(kp)) {
            if (m_kwargs.contains(k.first)) {
#if defined(NDEBUG)
                multiple_values_error();
#else
                multiple_values_error(str(k.first));
#endif
            }
            m_kwargs[k.first] = k.second;
        }
    }

    [[noreturn]] static void nameless_argument_error() {
        throw type_error("Got kwargs without a name; only named arguments "
                         "may be passed via py::arg() to a python function call. "
                         "(compile in debug mode for details)");
    }
    [[noreturn]] static void nameless_argument_error(std::string type) {
        throw type_error("Got kwargs without a name of type '" + type + "'; only named "
                         "arguments may be passed via py::arg() to a python function call. ");
    }
    [[noreturn]] static void multiple_values_error() {
        throw type_error("Got multiple values for keyword argument "
                         "(compile in debug mode for details)");
    }

    [[noreturn]] static void multiple_values_error(std::string name) {
        throw type_error("Got multiple values for keyword argument '" + name + "'");
    }

    [[noreturn]] static void argument_cast_error() {
        throw cast_error("Unable to convert call argument to Python object "
                         "(compile in debug mode for details)");
    }

    [[noreturn]] static void argument_cast_error(std::string name, std::string type) {
        throw cast_error("Unable to convert call argument '" + name
                         + "' of type '" + type + "' to Python object");
    }

private:
    tuple m_args;
    dict m_kwargs;
};

/// Collect only positional arguments for a Python function call
template <return_value_policy policy, typename... Args,
          typename = enable_if_t<all_of<is_positional<Args>...>::value>>
simple_collector<policy> collect_arguments(Args &&...args) {
    return simple_collector<policy>(std::forward<Args>(args)...);
}

/// Collect all arguments, including keywords and unpacking (only instantiated when needed)
template <return_value_policy policy, typename... Args,
          typename = enable_if_t<!all_of<is_positional<Args>...>::value>>
unpacking_collector<policy> collect_arguments(Args &&...args) {
    // Following argument order rules for generalized unpacking according to PEP 448
    static_assert(
        constexpr_last<is_positional, Args...>() < constexpr_first<is_keyword_or_ds, Args...>()
        && constexpr_last<is_s_unpacking, Args...>() < constexpr_first<is_ds_unpacking, Args...>(),
        "Invalid function call: positional args must precede keywords and ** unpacking; "
        "* unpacking must precede ** unpacking"
    );
    return unpacking_collector<policy>(std::forward<Args>(args)...);
}

template <typename Derived>
template <return_value_policy policy, typename... Args>
object object_api<Derived>::operator()(Args &&...args) const {
    return detail::collect_arguments<policy>(std::forward<Args>(args)...).call(derived().ptr());
}

template <typename Derived>
template <return_value_policy policy, typename... Args>
object object_api<Derived>::call(Args &&...args) const {
    return operator()<policy>(std::forward<Args>(args)...);
}

NAMESPACE_END(detail)

#define PYBIND11_MAKE_OPAQUE(Type) \
    namespace pybind11 { namespace detail { \
        template<> class type_caster<Type> : public type_caster_base<Type> { }; \
    }}

NAMESPACE_END(PYBIND11_NAMESPACE)
