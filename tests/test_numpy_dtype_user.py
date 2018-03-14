import sys
import pytest
from pybind11_tests import ConstructorStats
from pybind11_tests import numpy_dtype_user as m

stats_c = ConstructorStats.get(m.Custom)
# stats_str = ConstructorStats.get(m.CustomStr)

def test_dtype_intance():
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

# sys.stdout = sys.stderr
sys.argv = [__file__, "-s"]
pytest.main(args=sys.argv[1:])
