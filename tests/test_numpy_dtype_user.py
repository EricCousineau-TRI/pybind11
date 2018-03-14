import gc
import sys
import numpy as np

import pytest
from pybind11_tests import ConstructorStats
from pybind11_tests import numpy_dtype_user as m

stats = ConstructorStats.get(m.Custom)
# stats_str = ConstructorStats.get(m.CustomStr)

def test_scalar_ctor():
    """ Tests a single scalar instance. """
    c = m.Custom()
    c1 = m.Custom(10)
    c2 = m.Custom(c1)
    assert id(c) != id(c1)
    assert id(c) != id(c2)
    del c
    del c1
    del c2
    pytest.gc_collect()
    print('wooh')
    assert stats.alive() == 0
 
def test_scalar_meta():
    assert issubclass(m.Custom, np.generic)
    assert isinstance(np.dtype(m.Custom), np.dtype)

def test_scalar_op():
    a = m.Custom(1)
    b = m.Custom(2)
    assert m.same(a, a)
    assert not m.same(a, b)

def test_array_creation():
    # Zeros.
    x = np.zeros((2, 2), dtype=m.Custom)
    assert x.shape == (2, 2)
    assert x.dtype == m.Custom
    # Generic creation.
    x = np.array([m.Custom(1)])
    assert x.dtype == m.Custom
    # - Limitation on downcasting.
    x = np.array([m.Custom(1), 1])
    assert x.dtype == object

def test_array_cast():
    x = np.array([m.Custom(1)])
    def check(dtype):
        dx = x.astype(dtype)
        assert dx.dtype == dtype, dtype
        assert dx.astype(m.Custom).dtype == m.Custom
    check(m.Custom)
    check(float)
    check(object)

def check_array(actual, expected):
    expected = np.array(expected)
    if actual.shape != expected.shape:
        return False
    for a, b in zip(actual.flat, expected.flat):
        if not m.same(a, b):
            return False
    if actual.dtype != expected.dtype:
        return False
    return True

def test_array_ufunc():
    x = np.array([m.Custom(4)])
    y = np.array([m.Custom(2)])
    assert check_array(x + y, [m.Custom(6)])
    assert check_array(x * y, [m.Custom(8)])
    assert check_array(x - y, [m.Custom(2)])
    assert check_array(-x, [m.Custom(-4)])
    assert check_array(x == y, [m.CustomStr("4 == 2")])
    assert check_array(x < y, [False])

# sys.stdout = sys.stderr
# sys.argv = [__file__, "-s"]
# pytest.main(args=sys.argv[1:])
def main():
    pytest.gc_collect = gc.collect
    test_scalar_ctor()
    test_scalar_meta()
    test_scalar_op()
    test_array_creation()
    test_array_cast()
    test_array_ufunc()

import trace
tracer = trace.Trace(
    ignoredirs=sys.path,
    trace=1,
    count=0)
tracer.run('main()')
