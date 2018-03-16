import gc
import sys
import numpy as np

import pytest
from pybind11_tests import ConstructorStats
from pybind11_tests import numpy_dtype_user as m

stats = ConstructorStats.get(m.Custom)

def check_array(actual, expected):
    """Checks if two arrays are exactly similar (shape, type, and data)."""
    expected = np.array(expected)
    if actual.shape != expected.shape:
        return False
    if not m.same(actual, expected).all():
        return False
    if actual.dtype != expected.dtype:
        return False
    return True

def test_scalar_ctor():
    """Tests instance lifetime management since we had to redo the instance
    registry to inherit from `np.generic` :( """
    c = m.Custom()
    c1 = m.Custom(c)
    assert id(c) != id(c1)
    assert id(c.self()) == id(c)
    del c
    del c1
    pytest.gc_collect()
    assert stats.alive() == 0
 
def test_scalar_meta():
    """Tests basic metadata."""
    assert issubclass(m.Custom, np.generic)
    assert isinstance(np.dtype(m.Custom), np.dtype)

def test_scalar_op():
    """Tests scalar operators."""
    a = m.Custom(1)
    b = m.Custom(2)
    assert m.same(a, a)
    assert not m.same(a, b)
    a += 2
    assert m.same(a, m.Custom(3))

def test_array_creation():
    # Zeros.
    x = np.zeros((2, 2), dtype=m.Custom)
    assert x.shape == (2, 2)
    assert x.dtype == m.Custom
    # Generic creation.
    x = np.array([m.Custom(1, "Howdy")])
    assert x.dtype == m.Custom
    # - Limitation on downcasting when mixing types.
    x = np.array([m.Custom(10), 1.])
    assert x.dtype == object
    # - At present, we will be leaking memory. This doesn't show up in instance
    # count, since these items are only mutated via `operator=`; however, we're
    # gonna be leaking.

def test_array_cast():
    def check(x, dtype):
        dx = x.astype(dtype)
        assert dx.dtype == dtype, dtype
        assert dx.astype(m.Custom).dtype == m.Custom
    # Custom -> {Custom, float, object}
    x = np.array([m.Custom(1)])
    check(x, m.Custom)
    check(x, float)
    check(x, object)
    # float -> Custom
    x = np.array([1., 2.])
    check(x, m.Custom)
    # object -> Custom
    # N.B. Only explicit casts are allowed here. Can't supply floats.
    # TODO(eric.cousineau): This *will* be annoying. See if there's a way
    # around this.
    x = np.array([m.Custom(10), m.SimpleStruct(100)], dtype=object)
    check(x, m.Custom)

def test_array_cast_implicit():
    a = np.array([1.]).astype(m.Custom)
    # - We registered `Custom{} + double{}`.
    a += 2
    assert check_array(a, [m.Custom(3.)])
    # - Try multiple elements.
    a = np.array([1., 2.]).astype(m.Custom)
    a += 2.
    assert check_array(a, [m.Custom(3.), m.Custom(4.)])
    # We do not allow implicit coercion for `double`:    
    with pytest.raises(TypeError):
        a[0] = 1.
    with pytest.raises(TypeError):
        b = np.array([1., 2.], dtype=m.Custom)
    with pytest.raises(TypeError):
        a *= 2
    # Try an implicit conversion.
    a = np.array([1.]).astype(m.Custom)
    # - Nominal pybind implicit conversion
    a[0] = m.SimpleStruct(9)
    assert check_array(a, [m.Custom(9)])
    # - Test array construction (numpy coercion)
    c = np.array([m.SimpleStruct(10), m.SimpleStruct(11)], dtype=m.Custom)
    assert check_array(c, [m.Custom(10), m.Custom(11)])
    # - Test implicit cast via coercion.
    c *= m.SimpleStruct(2)
    assert check_array(c, [m.Custom(20), m.Custom(22)])
    # - Show dangers of implicit conversion (info loss).
    ds = m.Custom(100, "Hello")
    d = np.array([ds])
    e = np.array([m.SimpleStruct(0)])
    e[:] = d
    d[:] = e
    assert not check_array(d, [ds])
    assert check_array(d, [m.Custom(100)])

def test_array_ufunc():
    x = np.array([m.Custom(4)])
    y = np.array([m.Custom(2, "World")])
    assert check_array(x + y, [m.Custom(6)])
    assert check_array(x * y, [m.Custom(8)])
    assert check_array(x - y, [m.Custom(2)])
    assert check_array(-x, [m.Custom(-4)])
    assert check_array(x == y, [m.CustomStr("4 == 2 && '' == 'World'")])
    assert check_array(x < y, [False])
    assert check_array(np.power(x, y), [m.CustomStr("4 ^ 2")])

sys.stdout = sys.stderr
def main():
    pytest.gc_collect = gc.collect
    # test_scalar_ctor()
    # test_scalar_meta()
    # test_scalar_op()
    # test_array_creation()
    test_array_cast()
    # test_array_cast_implicit()
    # test_array_ufunc()

import trace
tracer = trace.Trace(trace=1, count=0, ignoredirs=sys.path)
tracer.run('main()')
