import os
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

prefer_user_copyswap = np and 'NUMPY_PATCH' in os.environ


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


def test_scalar_functions():
    # Test scalar functions.
    a = m.Custom(1)
    assert repr(a) == "Custom(1.0, '')"
    assert str(a) == "C<1.0, ''>"
    assert m.same(a, a)
    b = m.Custom(2)
    assert not m.same(a, b)


def test_scalar_algebra():
    """Tests scalar algebra."""
    a = m.Custom(1)
    b = m.Custom(2)
    # N.B. Implicit casting is not easily testable here; see array tests.
    # Operators.
    # - self + self
    assert m.same(a + b, m.Custom(3))
    a += b
    assert m.same(a, m.Custom(3))
    # - self + double (and int, implicitly)
    assert m.same(a + 2, m.Custom(5))
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
        # Fails due to `np.copyto` relying on `np.long` conversion on uninitialized memory.
        x = np.ones((2, 2), dtype=m.Custom)
    x = np.ones((1, 2)).astype(m.Custom)
    assert check_array(x, [[m.Custom(1), m.Custom(1)]])
    x = np.full((1, 2), m.Custom(10), dtype=m.Custom)
    assert check_array(x, [[m.Custom(10), m.Custom(10)]])
    # `np.eye(..., dtype=m.Custom)` requires a converter from `int` to
    # `m.Custom` (or something that tells it to use `double`).
    # Prefer to avoid, as it complicates other implicit conversions,
    # which in general shouldn't be there, but nonetheless should be tested.
    x = np.eye(2).astype(m.Custom)
    assert check_array(x, [[m.Custom(1), m.Custom(0)], [m.Custom(0), m.Custom(1)]])


def check_array(actual, expected):
    """Checks if two arrays are exactly similar (shape, type, and data)."""
    expected = np.array(expected)
    if actual.shape != expected.shape:
        return False
    if actual.dtype != expected.dtype:
        return False
    if not m.same(actual, expected).all():
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
    # We do not allow implicit coercion from `double` to Custom:
    with pytest.raises(TypeError):
        a[0] = 1.
    with pytest.raises(TypeError):
        b = np.array([1., 2.], dtype=m.Custom)
        print(b)  # Supress "unused" warnings
    with pytest.raises(TypeError):
        a *= 2
    # Try coercion from `Custom` to `double`
    af = np.array([0., 0.], dtype=float)
    # The following is viewed as "explicit" since we're using a slice, I guess?
    af[:] = a
    assert check_array(af, [3., 4.])
    af[:] = af + a
    assert check_array(af, [6., 8.])
    # This, I guess, tries explicit coercion, which is not available.
    with pytest.raises(TypeError):
        af += a
    # Also causes an error, as it attempts implicit coercion, which we've
    # disabled.
    with pytest.raises(TypeError):
        af[0] = a[0]
    # - Try an unregistered type, e.g. `int`.
    ai = np.array([0, 0], dtype=int)
    with pytest.raises(ValueError):
        ai[:] = a
    with pytest.raises(TypeError):
        ai += a
    with pytest.raises(TypeError):
        ai[0] = a[0]
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
    assert check_array(x + 1, [m.Custom(5)])
    assert check_array(1 + x, [m.Custom(5)])
    assert check_array(x * y, [m.Custom(8)])
    assert check_array(x - y, [m.Custom(2)])
    assert check_array(x / y, [m.Custom(2)])
    assert check_array(-x, [m.Custom(-4)])
    assert check_array(x == y, [m.CustomStr("4 == 2 && '' == 'World'")])
    assert check_array(x < y, [False])
    assert check_array(np.power(x, y), [m.CustomStr("4 ^ 2")])
    assert check_array(np.dot(x, y), m.Custom(8))
    assert check_array(np.dot([x], [y]), [[m.Custom(8)]])
    assert check_array(np.cos(x), [m.Custom(np.cos(x[0].value()))])
    assert check_array(np.logical_and(x, y), [10.])


def test_array_op_order():
    s = m.SimpleStruct(1)
    f = 0.
    assert f + s == -1
    assert s + s == 0
    assert s + f == 1
    sv = np.array([s])
    fv = np.array([f])
    assert check_array(sv + sv, [0.])
    assert check_array(sv + fv, [1.])
    assert check_array(fv + sv, [-1.])


def test_object_mixing():
    # Ensure that operations involving `dtype=object` downcasts properly.
    c = m.Custom(0)
    a = m.ObjectA()
    b = m.ObjectB()
    cv = np.array([c])
    av = np.array([a])
    bv = np.array([b])
    # These operations are now downcast to `object`.
    assert check_array(cv + av, np.array([m.Custom(1000.)], dtype=object))
    assert check_array(cv + bv, np.array([m.Custom(9999.)], dtype=object))


def test_implicit_arguments():
    s1 = m.SimpleStruct(1)
    s2 = m.SimpleStruct(1000)
    s1a = np.array(s1)
    s2a = np.array(s2)
    y = m.binary_op(s1, s2)
    assert m.same(y, m.CustomStr("1 == 1000"))
    y = m.binary_op(s1a, s2a)
    assert m.same(y, m.CustomStr("1 == 1000"))
    with pytest.raises(TypeError):
        # This does not work, even when declaring implicit arguments. Most likely
        # because NumPy needs to know an anchoring type?
        y = m.binary_op_loop(s1a, s2a)
    # The following works because NumPy is aware of the type...
    c1 = m.Custom(s1)
    y = m.binary_op_loop(c1, s2a)
    assert m.same(y, m.CustomStr("1 == 1000"))


def test_reference_arguments():
    # Test Python -> C++ reference
    x = np.array([m.Custom(0), m.Custom(1)])
    m.add_one(x)
    assert check_array(x, [m.Custom(1), m.Custom(2)])
    # Test C++ -> Python reference
    c = m.Container()
    xc = c.value()
    assert xc.shape == (1, 2)
    xc += 10
    assert check_array(c.value(), [
        [m.Custom(10), m.Custom(11)]])
    m.add_one(c.value())
    assert check_array(c.value(), [
        [m.Custom(11), m.Custom(12)]])


@pytest.mark.skipif(not prefer_user_copyswap, reason="requires NumPy patch")
def test_copy():
    x = np.array([m.Custom(1, "a")])
    y = np.copy(x)
    x[0] = m.Custom(10, "c")
    assert x[0].value() == 10
    assert x[0].str() == "c"
    assert y[0].value() == 1
    assert y[0].str() == "a"


def test_implicit_arg():
    # Test implicit casting from an integer.
    x = m.implicit_arg_scalar(1)
    assert isinstance(x, m.ImplicitArg)
    iv = np.array([1, 2])
    assert iv.dtype == np.int64
    xv = m.implicit_arg_vector(iv)
    assert xv.dtype == m.ImplicitArg
    assert xv.shape == (2,)
    m.implicit_arg_vector([1, 2])
    m.implicit_arg_vector([1., 2.])
    m.implicit_arg_vector(np.array([1., 2.]))


def test_result_type():
    dt = np.result_type(m.ImplicitArg, np.int64)
    assert dt == m.ImplicitArg
    dt = np.result_type(m.ImplicitArg, np.float)
    assert dt == m.ImplicitArg
    dt = np.result_type(m.ImplicitArg, 1.)
    assert dt == m.ImplicitArg
