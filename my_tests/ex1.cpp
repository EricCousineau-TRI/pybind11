#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include "Eigen/Core"
#include "unsupported/Eigen/AutoDiff"

#include <pybind11/embed.h>

#define ARRAY 0
typedef Eigen::AutoDiffScalar<Eigen::VectorXd> AD;
typedef const Eigen::Matrix<double, 2, 2> PlainMat;
typedef Eigen::VectorXd PlainVec;
typedef Eigen::Matrix<AD, 2, 2> ADMat;
//typedef Eigen::Ref<Eigen::VectorXd> PlainVec;
#if WITH_AD == 1
//typedef Eigen::Matrix<AD, 4, 1> MyType;
typedef Eigen::Matrix<AD, Eigen::Dynamic, 1> MyType;
#else
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> MyType;
//typedef Eigen::Matrix<double, 2, 2> MyType;
#endif

namespace basics {
class MyClass{
 public:
  MyClass() {};
  MyType mat; // = {1, 2, 3, 4};
  MyType &getADScalarMatrix() {
	  mat << 1 , 2, 3, 4;
	  return mat;
  }
  MyType getSum(const pybind11::EigenDRef<const MyType> &A, const pybind11::EigenDRef<const MyType> &B) {
	  MyType C;
	  std::cout << "A = " << A << std::endl;
	  std::cout << "B = " << B << std::endl;
//	  std::cout << "A's address in getSum: " << &A <<std::endl;
//	  std::cout << "B's address in getSum: " << &B <<std::endl;
	  C = A + B;
//	  A = A + B;
	  std::cout << "C = " << C << std::endl;
	  return C;
  }

  MyType getSumExpr(const Eigen::Ref<const MyType> &A, const Eigen::Ref<const MyType> &B){
	  MyType C;
	  C = A + B;
	  std::cout << "A = " << A << std::endl;
	  std::cout << "B = " << B << std::endl;
	  std::cout << "C = " << C << std::endl;
	  return C;
  }

  void getSumInA(MyType &A, MyType &B) {
//	  std::cout << "A's address in getSumInA : " << &A <<std::endl;
//	  std::cout << "B's address in getSumInA : " << &B <<std::endl;
	  A << 100, 100, 100, 100;
	  std::cout << "A = " << A << std::endl;
  }
  void getPlainNumpySum(PlainMat& A, PlainMat& B) {
	  //				return (A + B);
  }
  void getADSum(ADMat& A, ADMat& B) {
	  //				return (A + B);
  }
 private:
};
}

namespace py = pybind11;
template <typename Derived>
AD eval(const Eigen::AutoDiffScalar<Derived>& x) {
	return AD(x.value(), x.derivatives());
}

#if WITH_AD == 1
namespace pybind11 {
namespace detail {
template <>
struct npy_format_descriptor<AD> {
 public:
    enum { value = npy_api::NPY_OBJECT_ };
            static pybind11::dtype dtype() { \
                if (auto ptr = npy_api::get().PyArray_DescrFromType_(value)) { \
                	auto tmp = reinterpret_borrow<pybind11::dtype>(ptr); \
                    return reinterpret_borrow<pybind11::dtype>(ptr); \
                }
                pybind11_fail("Unsupported buffer format!"); \
    pybind11_fail("Unsupported buffer format!");
  	}
  static constexpr auto name = _("object"); \
};
}
};
#endif

PYBIND11_MODULE(basics, m) {
		m.doc() = "pybind11 basics module";

		py::class_<AD>(m, "AutoDiffXd")
		.def("__init__",
		[](AD& self,
		double value,
		const Eigen::VectorXd& derivatives) {
			new (&self) AD(value, derivatives);
		})
		.def("value", [](const AD& self) {
			return self.value();
		});

		py::class_<basics::MyClass>(m, "MyClass")
		.def(py::init())
#if WITH_AD == 1
		.def_readwrite("mat", &basics::MyClass::mat)
		.def("getADScalarMatrix", &basics::MyClass::getADScalarMatrix, py::return_value_policy::reference_internal)
		//		MyType getSum(Eigen::Ref<MyType> &A, Eigen::Ref<MyType> &B) {
		//			.def("getSum", [](basics::MyClass::getSum)
		.def("getADSum",[](basics::MyClass& myclass, ADMat& x, ADMat& y){myclass.getADSum(x,y);})
#endif
		.def("getSum", [](basics::MyClass& myclass, pybind11::EigenDRef<MyType> &A, pybind11::EigenDRef<MyType> &B)
		{return myclass.getSum(A, B);})
		.def("getSumExpr", [](basics::MyClass& myclass, const Eigen::Ref<const MyType> &A, const Eigen::Ref<const MyType> &B)
		{return myclass.getSumExpr(A, B);})

		.def("getSumInA", &basics::MyClass::getSumInA)
		.def("getPlainNumpySum",[](basics::MyClass& myclass, PlainMat& x, PlainMat& y){myclass.getPlainNumpySum(x,y);})
		;
}


int main(int argc, char** argv) {
  std::cout << "Executing..." << std::endl;
  {
    py::scoped_interpreter guard{};
    py::module m("basics");
    pybind11_init_basics(m);
    py::globals()["basics"] = m;

    // Pass in command-line arguments?
    std::string file = "externals/pybind11/my_tests/my_python.py";
    py::object sys = py::module::import("sys");
    py::list py_argv = py::list();
    py_argv.append(file);
    for (int i = 1; i < argc; ++i)
      py_argv.append(argv[i]);
    sys.attr("argv") = py_argv;

    py::eval_file(file.c_str());
  }

  return 0;
}
