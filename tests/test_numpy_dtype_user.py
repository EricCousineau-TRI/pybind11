import pytest
from pybind11_tests import ConstructorStats
from pybind11_tests import numpy_dtype_user as m


def test_dtype_intance():
    """ Tests a single scalar instance. """
    cstats = ConstructorStats.get(
    n_inst = ConstructorStats.detail_reg_inst()
