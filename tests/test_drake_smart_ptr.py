# -*- coding: utf-8 -*-
import pytest
import sys
import weakref

import env  # noqa: F401

from pybind11_tests import drake_smart_ptr as m
from pybind11_tests import ConstructorStats


def test_pointer_caster():
    assert m.test_pointer_caster()


@pytest.mark.skip(
    reason="Generally reproducible in CPython, Python 3, non-debug, on Linux. "
    "However, hard to pin this down for CI."
)
def test_1922():
    # Test #1922 (drake#11424).
    # Define a derived class which *does not* overload the method.
    # WARNING: The reproduction of this failure may be platform-specific, and
    # seems to depend on the order of definition and/or the name of the classes
    # defined. For example, trying to place this and the C++ code in
    # `test_virtual_functions` makes `assert id_1 == id_2` below fail.
    class Child1(m.ExampleVirt2):
        pass

    id_1 = id(Child1)
    assert m.example_virt2_get_name(m.ExampleVirt2()) == "ExampleVirt2"
    assert m.example_virt2_get_name(Child1()) == "ExampleVirt2"

    # Now delete everything (and ensure it's deleted).
    wref = weakref.ref(Child1)
    del Child1
    pytest.gc_collect()
    assert wref() is None

    # Define a derived class which *does* define an overload.
    class Child2(m.ExampleVirt2):
        def get_name(self):
            return "Child2"

    id_2 = id(Child2)
    assert id_1 == id_2  # This happens in CPython; not sure about PyPy.
    assert m.example_virt2_get_name(m.ExampleVirt2()) == "ExampleVirt2"
    # THIS WILL FAIL: This is using the cached `ExampleVirt2.get_name`, rather
    # than re-inspect the Python dictionary.
    assert m.example_virt2_get_name(Child2()) == "Child2"


@pytest.mark.skipif(env.PYPY, reason="Unsupported on PyPy")
def test_mi_ownership_constraint():
    # See below for positive tests.

    # unique_ptr
    c = m.ContainerBase1(m.MIType(10, 100))
    assert c is not None

    # shared_ptr
    # Should not throw an error.
    obj = m.Base12a(10, 100)
    assert obj.bar() == 100
    c = m.ContainerBase2a(obj)
    # Use variable to avoid style violations.
    assert c is not None

    # TODO(eric.cousineau): This currently causes a segfault in both
    # this branch and on `master` (a303c6f).
    # Figure out if there is a way to fix this?

    # assert c.get().bar() == 100

    # # Should throw an error.
    # c = m.ContainerBase2a(m.Bae12a(10, 100))
    # assert c.get().foo() == 10
    # assert c.get().bar() == 100


def define_child(name, base_type, stats_type):
    # Derived instance of `DefineBase<>` in C++.
    # `stats_type` is meant to enable us to use `ConstructorStats` exclusively for a Python class.

    class ChildT(base_type):
        def __init__(self, *args):
            base_type.__init__(self, *args)
            self.icstats = m.get_instance_cstats(ChildT.get_cstats(), self)
            self.icstats.track_created()

        def __del__(self):
            self.icstats.track_destroyed()

        def value(self):
            # Use a different value, so that we can detect slicing.
            return 10 * base_type.value(self)

        @staticmethod
        def get_cstats():
            return ConstructorStats.get(stats_type)

    ChildT.__name__ = name
    return ChildT


ChildBad = define_child("ChildBad", m.BaseBad, m.ChildBadStats)
Child = define_child("Child", m.Base, m.ChildStats)

ChildBadUnique = define_child("ChildBadUnique", m.BaseBadUnique, m.ChildBadUniqueStats)
ChildUnique = define_child("ChildUnique", m.BaseUnique, m.ChildUniqueStats)


# TODO(eric.cousineau): See if this is at all possibly on PyPy.
# Placing `pytest.gc_collect` near `del` statements indicates that we are not
# capturing deletion properly.
@pytest.mark.skipif(env.PYPY, reason="Unsupported on PyPy")
def test_shared_ptr_derived_slicing(capture):
    leaked_count = [0]
    is_py38 = sys.version_info[:2] >= (3, 8)

    def py38_leak():
        if is_py38:
            leaked_count[0] += 1

    def cstats_alive_except_leaked():
        return cstats.alive() - leaked_count[0]

    # [ Bad ]
    cstats = ChildBad.get_cstats()
    # Create instance in move container to permit releasing.
    obj = ChildBad(10)
    obj_weak = weakref.ref(obj)
    # This will release the reference, the refcount will drop to zero, and Python will destroy it.
    c = m.BaseBadContainer(obj)
    del obj
    # We will have lost the derived Python instance.
    assert obj_weak() is None
    # Check stats:
    assert cstats.alive() == 0
    # As an additional check, we will try to query the value from the container's value.
    # This should have been 100 if the trampoline had retained its Python portion.
    assert c.get().value() == 10
    # Destroy references.
    del c

    # [ Good ]
    # See above for setup.
    cstats = Child.get_cstats()
    # Try a temporary setup.
    c = m.BaseContainer(Child(10))
    assert cstats.alive() == 1
    obj = c.release()
    del c
    assert cstats.alive() == 1
    assert obj.value() == 100
    del obj
    py38_leak()
    assert cstats_alive_except_leaked() == 0
    # Use something more permanent.
    obj = Child()  # Use factory method.
    obj_weak = weakref.ref(obj)
    assert cstats_alive_except_leaked() == 1
    c = m.BaseContainer(obj)
    del obj
    assert cstats_alive_except_leaked() == 1
    # We now still have a reference to the object. py::wrapper<> will intercept Python's
    # attempt to destroy `obj`, is aware the `shared_ptr<>.use_count() > 1`, and will increase
    # the ref count by transferring a new reference to `py::wrapper<>` (thus reviving the object,
    # per Python's documentation of __del__).
    assert obj_weak() is not None
    assert cstats_alive_except_leaked() == 1
    # This goes from C++ -> Python, and then Python -> C++ once this statement has finished.
    assert c.get().value() == 100
    assert cstats_alive_except_leaked() == 1
    # Destroy references (effectively in C++), and ensure that we have the desired behavior.
    del c
    py38_leak()
    assert cstats_alive_except_leaked() == 0

    # Ensure that we can pass it from Python -> C++ -> Python, and ensure that C++ does not think
    # that it has ownership.
    obj = Child(10)
    c = m.BaseContainer(obj)
    del obj
    assert cstats_alive_except_leaked() == 1
    obj = c.get()
    # Now that we have it in Python, there should only be 1 Python reference, since
    # py::wrapper<> in C++ should have released its reference.
    expected_refcount = 2
    if is_py38:
        expected_refcount += 1
    assert sys.getrefcount(obj) == expected_refcount
    del c
    assert cstats_alive_except_leaked() == 1
    assert obj.value() == 100
    del obj
    py38_leak()
    assert cstats_alive_except_leaked() == 0


@pytest.mark.skipif(env.PYPY, reason="Unsupported on PyPy")
def test_unique_ptr_derived_slicing(capture):
    # [ Bad ]
    cstats = ChildBadUnique.get_cstats()
    # Create instance in move container to permit releasing.
    try:
        c = m.BaseBadUniqueContainer(ChildBadUnique(10))
    except RuntimeError as e:
        assert "wrapper<>" in str(e)
    # We lost this portion to slicing.
    assert cstats.alive() == 0

    # [ Good ]
    cstats = ChildUnique.get_cstats()
    # Try a temporary setup.
    c = m.BaseUniqueContainer(ChildUnique(10))
    assert cstats.alive() == 1
    obj = c.release()
    del c
    assert cstats.alive() == 1
    assert obj.value() == 100
    del obj
    assert cstats.alive() == 0

    # Ensure that we can pass between Python -> C++ -> Python
    # (using factory method).
    obj = m.BaseUniqueContainer(ChildUnique()).release()
    obj = m.BaseUniqueContainer(obj).release()
    assert obj.value() == 100
    del obj
    assert cstats.alive() == 0


def test_unique_ptr_arg():
    stats = ConstructorStats.get(m.UniquePtrHeld)

    pass_through_list = [
        m.unique_ptr_pass_through,
        m.unique_ptr_pass_through_move_to_py,
        m.unique_ptr_pass_through_cast_to_py,
        # TODO(eric.cousineau): Fix these cases.
        # m.unique_ptr_pass_through_cast_from_py,
        # m.unique_ptr_pass_through_move_from_py,
    ]
    for pass_through in pass_through_list:
        obj = m.UniquePtrHeld(1)
        obj_ref = pass_through(obj)
        assert stats.alive() == 1
        assert obj.value() == 1
        assert obj == obj_ref
        del obj
        del obj_ref
        pytest.gc_collect()
        assert stats.alive() == 0

    obj = m.UniquePtrHeld(1)
    m.unique_ptr_terminal(obj)
    assert stats.alive() == 0

    m.unique_ptr_terminal(m.UniquePtrHeld(2))
    assert stats.alive() == 0

    assert m.unique_ptr_pass_through(None) is None
    m.unique_ptr_terminal(None)

    with pytest.raises(TypeError):
        m.unique_ptr_terminal(m.UniquePtrOther())


def test_unique_ptr_to_shared_ptr():
    obj = m.shared_ptr_held_in_unique_ptr()
    assert m.shared_ptr_held_func(obj)


def test_unique_ptr_overload_fail():
    obj = m.UniquePtrHeld(1)
    # These overloads pass ownership back to Python.
    out = m.unique_ptr_overload(obj, m.FirstT())
    assert out["obj"] is obj
    assert out["overload"] == 1
    out = m.unique_ptr_overload(obj, m.SecondT())
    assert out["obj"] is obj
    assert out["overload"] == 2


def test_unique_ptr_held_container_from_cpp():
    def check_reset(obj_new):
        c = m.UniquePtrHeldContainer()
        obj = c.get()
        assert obj is not None
        obj_ref = c.reset(obj_new)
        assert obj is obj_ref
        assert c.get() is obj_new

    check_reset(obj_new=m.UniquePtrHeld(10))
    check_reset(obj_new=None)
