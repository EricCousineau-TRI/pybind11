import pytest
from pybind11_tests import ConstructorStats
from pybind11_tests import numpy_dtype_user as m

stats_c = ConstructorStats.get(m.Custom)
# stats_str = ConstructorStats.get(m.CustomStr)

def test_dtype_intance():
    """ Tests a single scalar instance. """
    print("wooh")
