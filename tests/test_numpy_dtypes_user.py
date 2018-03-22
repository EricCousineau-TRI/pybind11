import pytest

from pybind11_tests import ConstructorStats
from pybind11_tests import numpy_dtype_user as m

np = None
with pytest.suppress(ImportError):
    import numpy as np
    import sys
    sys.stderr.write("numpy version: {} {}\n".format(
        np.version.full_version, np.version.git_revision))

pytestmark = pytest.mark.skipif(
    not np or hasattr(m, "DISABLED"), reason="requires numpy and C++ >= 14")

def test_scalar_meta():
    """Tests basic metadata."""
    assert issubclass(m.Custom, np.generic)
    assert isinstance(np.dtype(m.Custom), np.dtype)


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
    stats = ConstructorStats.get(m.Custom)
    assert stats.alive() == 0


def test_scalar_op():
    """Tests scalar operators."""
    a = m.Custom(1)
    assert repr(a) == "Custom(1.0, '')"
    assert str(a) == "C<1.0, ''>"
    assert m.same(a, a)
    b = m.Custom(2)
    assert not m.same(a, b)
    # Implicit casting is not easily testable here; see array tests.
    # Operators.
    # - self + self
    assert m.same(a + b, m.Custom(3))
    a += b
    assert m.same(a, m.Custom(3))
    # - self + double (and int, implicitly)
    assert m.same(a + 2, m.Custom(5))
    print(2 + a)
    assert m.same(2 + a, m.Custom(5))
    a += 2.
    a += 1
    assert m.same(a, m.Custom(6))
    a = m.Custom(6)
    # Others.
    assert m.same(a * b, m.Custom(12))
    assert m.same(a - b, m.Custom(4))
    assert m.same(-a, m.Custom(-6))
    # Logical.
    assert m.same(a == b, m.CustomStr("6 == 2 && '' == ''"))
    assert m.same(a < b, False)


def test_array_creation():
    # Zeros.
    x = np.zeros((2, 2), dtype=m.Custom)
    assert x.shape == (2, 2)
    assert x.dtype == m.Custom
    # Generic creation.
    x = np.array([m.Custom(1, "Howdy")])
    assert x.dtype == m.Custom
    # - Limitation on downcasting when mixing types.
    # This could be alleviated by allowing doubles to be implicitly casted for
    # the type, but it's best to avoid that.
    x = np.array([m.Custom(10), 1.])
    assert x.dtype == object
    # - At present, we will be leaking memory. This doesn't show up in instance
    # count, since these items are only mutated via `operator=`; however, it will
    # still be the case for resizing.
    # See https://github.com/numpy/numpy/issues/10721 for more information.


def test_array_creation_extended():
    with pytest.raises(ValueError):
        # Fails due to shenanigans with `np.copyto`.
        x = np.ones((2, 2), dtype=m.Custom)
    x = np.ones((1, 2)).astype(m.Custom)
    assert check_array(x, [[m.Custom(1), m.Custom(1)]])
    x = np.full((1, 2), m.Custom(10), dtype=m.Custom)
    assert check_array(x, [[m.Custom(10), m.Custom(10)]])
    # `np.eye(..., dtype=m.Custom)` requires a converter from `int` to `m.Custom` (or something that
    # tells it to use `double`). Prefer to avoid, as it complicates other implicit conversions,
    # which in general shouldn't be there, but nonetheless should be tested.
    x = np.eye(2).astype(m.Custom)
    assert check_array(x, [[m.Custom(1), m.Custom(0)], [m.Custom(0), m.Custom(1)]])


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


def test_array_cast():
    def check(x, dtype):
        dx = x.astype(dtype)
        assert dx.dtype == dtype, dtype
        assert dx.astype(m.Custom).dtype == m.Custom
        return dx
    # Custom -> {Custom, float, object}
    x = np.array([m.Custom(1)])
    check(x, m.Custom)
    check(x, float)
    check(x, object)
    # float -> Custom
    x = np.array([1., 2.])
    check(x, m.Custom)
    # object -> Custom
    # - See notes in the C++ code for defining the ufunc cast for `object` to
    # `Class`.
    x = np.array([1., m.Custom(10), m.SimpleStruct(100)], dtype=object)
    dx = check(x, m.Custom)
    assert check_array(dx, [m.Custom(1), m.Custom(10), m.Custom(100)])


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
    # assert check_array(x + 1, [m.Custom(5)])
    assert check_array(x * y, [m.Custom(8)])
    assert check_array(x - y, [m.Custom(2)])
    assert check_array(-x, [m.Custom(-4)])
    assert check_array(x == y, [m.CustomStr("4 == 2 && '' == 'World'")])
    assert check_array(x < y, [False])
    assert check_array(np.power(x, y), [m.CustomStr("4 ^ 2")])
    assert check_array(np.dot(x, y), m.Custom(8))
    assert check_array(np.dot([x], [y]), [[m.Custom(8)]])
    print(x[0].cos())
    print(np.cos(x))


sys.stdout = sys.stderr
