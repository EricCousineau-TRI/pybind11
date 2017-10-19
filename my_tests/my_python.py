import numpy as np
from basics import *
import pdb
import sys

print(sys.argv)
WITH_AD = int(sys.argv[1])
APPLY_PDB = 0
WITH_ARRAY = 0

def get_sum_in_A():
    print "mat_1 before = ", mat1
    print "mat1[1][1].value()", mat1[1][1].value()
    my_class.getSumInA(mat1, mat2)
    print "mat_1 after = ", mat1
    print "mat1[1][1].value()", mat1[1][1].value()

def get_sum():
    if WITH_AD:
        if WITH_ARRAY:
            print(my_class.getSum(arr1, arr2))
        else:
            print(my_class.getSum(mat1, mat2))
            print "mat1 strides " + str(mat1.strides)
    else:
        my_class.getSum(a, b)
     #   my_class.getSumInA(A, B)
        # np.testing.assert_array_equal(a, 2*np.ones((2,2)))
        # np.testing.assert_array_equal(A, np.ones((2,2)))

def get_sum_expr():
    if WITH_AD:
        my_class.getSumExpr(arr1, arr2)
  #      my_class.getSumExpr(mat1, mat2)
    else:
        pdb.set_trace()
        my_class.getSumExpr(a, b)
     #   my_class.getSumExpr(A, B)


if(__name__ == "__main__"):
    my_class = MyClass()
    if WITH_AD:
        arr1 = np.array([AutoDiffXd(1, np.ones(1)),  AutoDiffXd(2,
                np.ones(1)), AutoDiffXd(3, np.ones(1)), AutoDiffXd(4, np.ones(1))])
        arr2 = np.array([AutoDiffXd(1, np.ones(1)),  AutoDiffXd(2,
                np.ones(1)), AutoDiffXd(3, np.ones(1)), AutoDiffXd(4, np.ones(1))])
        mat1 = np.array([[AutoDiffXd(1, np.ones(1)),  AutoDiffXd(2, np.ones(1))],
                [AutoDiffXd(3, np.ones(1)), AutoDiffXd(4, np.ones(1))]])
        mat2 = np.array([[AutoDiffXd(1, np.ones(1)),  AutoDiffXd(2,
                np.ones(1))], [AutoDiffXd(3, np.ones(1)), AutoDiffXd(4, np.ones(1))]])

    a = np.array([1,2,3,4])
    b = np.array([1,2,3,4])
    A = np.ones((2, 2))
    B = np.ones((2, 2))

    #my_class.getPlainNumpySum(a, b)
    #my_class.getADSum(mat1,mat2)
    #get_sum()
    get_sum_expr()
    #get_sum_in_A()
    #if APPLY_PDB: pdb.set_trace()
