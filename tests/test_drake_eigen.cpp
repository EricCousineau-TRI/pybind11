#include "pybind11_tests.h"
#include <pybind11/eigen.h>

#include <unsupported/Eigen/AutoDiff>
#include "Eigen/src/Core/util/DisableStupidWarnings.h"

typedef Eigen::AutoDiffScalar<Eigen::VectorXd> ADScalar;

template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> VectorXADScalar;
typedef Eigen::Matrix<ADScalar, 1, Eigen::Dynamic> VectorXADScalarR;
typedef Eigen::Matrix<ADScalar, 5, 1> Vector5ADScalar;
typedef Eigen::Matrix<ADScalar, 1, 6> Vector6ADScalarR;
PYBIND11_NUMPY_OBJECT_DTYPE(ADScalar);

using DenseADScalarMatrixR = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DenseADScalarMatrixC = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic>;

VectorXADScalar& get_cm_adscalar() {
    static VectorXADScalar value(1);
    return value;
};
VectorXADScalarR& get_rm_adscalar() {
    static VectorXADScalarR value(1);
    return value;
};

TEST_SUBMODULE(drake_general, m) {
    m.def("double_adscalar_col", [](const VectorXADScalar &x) -> VectorXADScalar { return 2.0f * x; });
    m.def("double_adscalar_col5", [](const Vector5ADScalar &x) -> Vector5ADScalar { return 2.0f * x; });
    m.def("double_adscalar_row", [](const VectorXADScalarR &x) -> VectorXADScalarR { return 2.0f * x; });
    m.def("double_adscalar_row6", [](const Vector6ADScalarR &x) -> Vector6ADScalarR { return 2.0f * x; });
    m.def("add_rm_adscalar", [](py::EigenDRef<VectorXADScalarR> x) { x.array() += 2; });
    m.def("add_cm_adscalar", [](py::EigenDRef<VectorXADScalar> x) { x.array() += 2; });
    m.def("get_cm_ref_adscalar", []() {
        return py::EigenDRef<VectorXADScalar>(get_cm_adscalar());
    });
    m.def("get_rm_ref_adscalar", []() {
        return py::EigenDRef<VectorXADScalarR>(get_rm_adscalar());
    });
    m.def("get_cm_const_ref_adscalar", []() { return Eigen::Ref<const VectorXADScalar>(get_cm_adscalar()); });
    m.def("get_rm_const_ref_adscalar", []() { return Eigen::Ref<const VectorXADScalarR>(get_rm_adscalar()); });

    // Increments ADScalar Matrix, returns a copy.
    m.def("incr_adscalar_matrix", [](const Eigen::Ref<const DenseADScalarMatrixC>& m, double v) {
      DenseADScalarMatrixC out = m;
      out.array() += v;
      return out;
    });

    // test_eigen_return_references, test_eigen_keepalive
    // return value referencing/copying tests:
    class ReturnTester {
        Eigen::MatrixXd mat = create();
        DenseADScalarMatrixR ad_mat = create_ADScalar_mat();
    public:
        ReturnTester() { print_created(this); }
        ~ReturnTester() { print_destroyed(this); }
        static Eigen::MatrixXd create() {  return Eigen::MatrixXd::Ones(10, 10); }
        static DenseADScalarMatrixR create_ADScalar_mat() { DenseADScalarMatrixR ad_mat(2, 2);
            ad_mat << 1, 2, 3, 7; return ad_mat; }
        static const Eigen::MatrixXd createConst() { return Eigen::MatrixXd::Ones(10, 10); }
        Eigen::MatrixXd &get() { return mat; }
        DenseADScalarMatrixR& get_ADScalarMat() {return ad_mat;}
        Eigen::MatrixXd *getPtr() { return &mat; }
        const Eigen::MatrixXd &view() { return mat; }
        const Eigen::MatrixXd *viewPtr() { return &mat; }
        Eigen::Ref<Eigen::MatrixXd> ref() { return mat; }
        Eigen::Ref<const Eigen::MatrixXd> refConst() { return mat; }
        Eigen::Block<Eigen::MatrixXd> block(int r, int c, int nrow, int ncol) { return mat.block(r, c, nrow, ncol); }
        Eigen::Block<const Eigen::MatrixXd> blockConst(int r, int c, int nrow, int ncol) const { return mat.block(r, c, nrow, ncol); }
        py::EigenDMap<Eigen::Matrix2d> corners() { return py::EigenDMap<Eigen::Matrix2d>(mat.data(),
                    py::EigenDStride(mat.outerStride() * (mat.outerSize()-1), mat.innerStride() * (mat.innerSize()-1))); }
        py::EigenDMap<const Eigen::Matrix2d> cornersConst() const { return py::EigenDMap<const Eigen::Matrix2d>(mat.data(),
                    py::EigenDStride(mat.outerStride() * (mat.outerSize()-1), mat.innerStride() * (mat.innerSize()-1))); }
    };
    using rvp = py::return_value_policy;
    py::class_<ReturnTester>(m, "ReturnTester")
        .def(py::init<>())
        .def_static("create", &ReturnTester::create)
        .def_static("create_const", &ReturnTester::createConst)
        .def("get", &ReturnTester::get, rvp::reference_internal)
        .def("get_ADScalarMat", &ReturnTester::get_ADScalarMat, rvp::reference_internal)
        .def("get_ptr", &ReturnTester::getPtr, rvp::reference_internal)
        .def("view", &ReturnTester::view, rvp::reference_internal)
        .def("view_ptr", &ReturnTester::view, rvp::reference_internal)
        .def("copy_get", &ReturnTester::get)   // Default rvp: copy
        .def("copy_view", &ReturnTester::view) //         "
        .def("ref", &ReturnTester::ref) // Default for Ref is to reference
        .def("ref_const", &ReturnTester::refConst) // Likewise, but const
        .def("ref_safe", &ReturnTester::ref, rvp::reference_internal)
        .def("ref_const_safe", &ReturnTester::refConst, rvp::reference_internal)
        .def("copy_ref", &ReturnTester::ref, rvp::copy)
        .def("copy_ref_const", &ReturnTester::refConst, rvp::copy)
        .def("block", &ReturnTester::block)
        .def("block_safe", &ReturnTester::block, rvp::reference_internal)
        .def("block_const", &ReturnTester::blockConst, rvp::reference_internal)
        .def("copy_block", &ReturnTester::block, rvp::copy)
        .def("corners", &ReturnTester::corners, rvp::reference_internal)
        .def("corners_const", &ReturnTester::cornersConst, rvp::reference_internal)
        ;

    py::class_<ADScalar>(m, "AutoDiffXd")
        .def("__init__",
             [](ADScalar & self,
                double value,
                const Eigen::VectorXd& derivatives) {
               new (&self) ADScalar(value, derivatives);
             })
        .def("value", [](const ADScalar & self) {
          return self.value();
        })
        .def("__repr__", [](const ADScalar& self) {
          return py::str("<ADScalar {} deriv={}>").format(self.value(), self.derivatives());
        })
        ;

    m.def("iss1105_col_obj", [](VectorXADScalar) { return true; });
    m.def("iss1105_row_obj", [](VectorXADScalarR) { return true; });
    m.def("cpp_matrix_shape", [](const MatrixX<ADScalar>& A) {
        return py::make_tuple(A.rows(), A.cols());
    });
    m.def("cpp_matrix_shape_ref", [](const Eigen::Ref<const MatrixX<ADScalar>>& A) {
        return py::make_tuple(A.rows(), A.cols());
    });
}
