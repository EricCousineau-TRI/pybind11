import pytest
from pybind11_tests import ownership_transfer as m
from pybind11_tests import ConstructorStats
from sys import getrefcount
import weakref

def define_child(base_cls, stat_cls):
    class child_cls(base_cls):
        @staticmethod
        def get_cstats():
            return ConstructorStats.get(stat_cls)

        def __init__(self, value):
            base_cls.__init__(self, value)
            self.value = value
            self.icstats = m.get_instance_cstats(child_cls.get_cstats(), self)
            self.icstats.track_created()

        def __del__(self):
            self.icstats.track_destroyed()

        def value(self):
            print("Child.value")
            return 10 * value
    return child_cls


ChildBad = define_child(m.BaseBad, m.ChildBadStats)
Child = define_child(m.Base, m.ChildStats)

def test_shared_ptr_derived_aliasing(capture):
    # [ Bad ]
    cstats = ChildBad.get_cstats()
    # Create instance in move container to permit releasing.
    obj = ChildBad(10)
    obj_weak = weakref.ref(obj)
    # This will release the reference, the refcount will drop to zero, and Python will destroy it.
    c = m.BaseBadContainer(obj)
    del obj
    # We will have lost the derived Python instance - it was garbage collected.
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
    obj = Child(10)
    obj_weak = weakref.ref(obj)
    c = m.BaseContainer(obj)
    del obj
    # assert iw() is not None
    assert cstats.alive() == 1
    assert c.get().value() == 100
    # Destroy references (effectively in C++), and ensure that we have the desired behavior.
    del c
    assert cstats.alive() == 0

if __name__ == "__main__":
    test_shared_ptr_derived_aliasing(None)
