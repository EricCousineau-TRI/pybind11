# -*- coding: utf-8 -*-
import pytest

import env  # noqa: F401

from pybind11_tests import drake_eigen as m


def float_to_adscalar(arr, deriv):
    arr = np.asarray(arr)
    assert arr.dtype == float
    new_arr = [m.AutoDiffXd(x, deriv) for x in arr.flat]
    return np.array(new_arr).reshape(arr.shape)


def adscalar_to_float(arr):
    arr = np.asarray(arr)
    assert arr.dtype == object
    new_arr = [x.value() for x in arr.flat]
    return np.array(new_arr).reshape(arr.shape)


def check_array(a, b):
    a, b = (np.asarray(x) for x in (a, b))
    assert a.shape == b.shape and a.dtype == b.dtype
    for index, (ai, bi) in enumerate(zip(a.flat, b.flat)):
        assert m.equal_to(ai, bi), index


def test_eigen_passing_adscalar():
    assert m.equal_to(1.0, 1.0)
    assert not m.equal_to(1.0, 1.1)
    assert m.equal_to(m.AutoDiffXd(0, [1.0]), m.AutoDiffXd(0, [1.0]))
    assert not m.equal_to(m.AutoDiffXd(0, [1.0]), m.AutoDiffXd(0, [1.1]))

    adscalar_mat = float_to_adscalar(ref, deriv=[1.0])
    adscalar_vec_col = adscalar_mat[:, 0]
    adscalar_vec_row = adscalar_mat[0, :]

    # Checking if a Python vector is getting doubled, when passed into a dynamic or fixed
    # row or col vector in Eigen.
    double_adscalar_mat = float_to_adscalar(2 * ref, deriv=[2.0])
    check_array(m.double_adscalar_col(adscalar_vec_col), double_adscalar_mat[:, 0])
    check_array(m.double_adscalar_col5(adscalar_vec_col), double_adscalar_mat[:, 0])
    check_array(m.double_adscalar_row(adscalar_vec_row), double_adscalar_mat[0, :])
    check_array(m.double_adscalar_row6(adscalar_vec_row), double_adscalar_mat[0, :])

    # Adding 7 to the a dynamic matrix using reference.
    incr_adscalar_mat = float_to_adscalar(ref + 7, deriv=[1.0])
    check_array(m.incr_adscalar_matrix(adscalar_mat, 7.0), incr_adscalar_mat)
    # The original adscalar_mat remains unchanged in spite of passing by reference, since
    # `Eigen::Ref<const CType>` permits copying, and copying is the only valid operation for
    # `dtype=object`.
    check_array(adscalar_to_float(adscalar_mat), ref)

    # Changes in Python are not reflected in C++ when internal_reference is returned.
    # These conversions should be disabled at runtime.

    def expect_ref_error(func):
        with pytest.raises(RuntimeError) as excinfo:
            func()
        assert "dtype=object" in str(excinfo.value)
        assert "reachable" not in str(excinfo.value)

    # - Return arguments.
    expect_ref_error(lambda: m.get_cm_ref_adscalar())
    expect_ref_error(lambda: m.get_rm_ref_adscalar())
    expect_ref_error(lambda: m.get_cm_const_ref_adscalar())
    expect_ref_error(lambda: m.get_rm_const_ref_adscalar())
    # - - Mutable lvalues referenced via `reference_internal`.
    return_tester = m.ReturnTester()
    expect_ref_error(lambda: return_tester.get_ADScalarMat())
    # - Input arguments, writeable `Ref<>`s.
    expect_ref_error(lambda: m.add_cm_adscalar(adscalar_vec_col))
    expect_ref_error(lambda: m.add_rm_adscalar(adscalar_vec_row))

    # Checking Issue 1105
    assert m.iss1105_col_obj(adscalar_vec_col[:, None])
    assert m.iss1105_row_obj(adscalar_vec_row[None, :])

    with pytest.raises(TypeError) as excinfo:
        m.iss1105_row_obj(adscalar_vec_col[:, None])
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        m.iss1105_col_obj(adscalar_vec_row[None, :])
    assert "incompatible function arguments" in str(excinfo.value)


def test_eigen_obj_shape():
    # Ensure that matrices are of the same shape for dtype=object when given a 1D NumPy array.
    # RobotLocomotion/drake#8620
    x_f = np.array([1, -1], dtype=float)
    shape_f = m.cpp_matrix_shape(x_f)
    x_ad = np.array([m.AutoDiffXd(0, []), m.AutoDiffXd(0, [])], dtype=object)
    shape_ad = m.cpp_matrix_shape(x_ad)
    shape_ad_ref = m.cpp_matrix_shape_ref(x_ad)
    assert shape_f == shape_ad
    assert shape_f == shape_ad_ref

