# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import keep_alive as m


def traced(func, ignoredirs=None):
    """Decorates func such that its execution is traced, but filters out any
     Python code outside of the system prefix."""
    # https://drake.mit.edu/python_bindings.html#debugging-with-the-python-bindings
    import functools
    import sys
    import trace
    if ignoredirs is None:
        ignoredirs = ["/usr", sys.prefix]
    tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return tracer.runfunc(func, *args, **kwargs)

    return wrapped


@traced
def test_stuff():
    other = m.ExampleOther()
    x = "a"

    out = m.example_free_func(other)
    assert isinstance(out, m.ExampleReturn)

    # Normal constructor.
    obj = m.ExampleSelf(other)
    # Factory constructors.
    m.ExampleSelf(other, str())
    m.ExampleSelf(other, int())

    out2 = obj.example_method(other)
    assert isinstance(out2, m.ExampleReturn)

    out3 = obj.example_static_method(other)
    assert isinstance(out3, m.ExampleReturn)

"""
Output, Ubuntu 18.04, using apt packages:

$ cd pybind11
$ mkdir build && cd build
$ cmake .. -DPYTHNO_EXECUTABLE=$(which python3)
$ make -j2 pytest
...
../../tests/test_keep_alive.py  --- modulename: test_keep_alive, funcname: test_stuff
test_keep_alive.py(27):     other = m.ExampleOther()
test_keep_alive.py(28):     x = "a"
test_keep_alive.py(30):     out = m.example_free_func(other)
keep_alive_impl( <ExampleReturn> , <ExampleOther> )
test_keep_alive.py(31):     assert isinstance(out, m.ExampleReturn)
test_keep_alive.py(34):     obj = m.ExampleSelf(other)
keep_alive_impl( <ExampleSelf> , <ExampleOther> )
test_keep_alive.py(36):     m.ExampleSelf(other, str())
keep_alive_impl( <ExampleSelf> , <ExampleOther> )
test_keep_alive.py(37):     m.ExampleSelf(other, int())
keep_alive_impl( None , <ExampleOther> )
test_keep_alive.py(39):     out2 = obj.example_method(other)
keep_alive_impl( <ExampleSelf> , <ExampleOther> )
test_keep_alive.py(40):     assert isinstance(out2, m.ExampleReturn)
test_keep_alive.py(42):     out3 = obj.example_static_method(other)
keep_alive_impl( <ExampleReturn> , <ExampleOther> )
test_keep_alive.py(43):     assert isinstance(out3, m.ExampleReturn)
"""
