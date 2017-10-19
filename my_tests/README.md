# Instructions

Execute:

    git clone https://github.com/EricCousineau-TRI/repro.git -b upstreaming_attempt_2
    cd repro 
    git submodule update --init -- externals/{eigen,pybind11}

    cd externals/pybind11/my_tests
    bazel build :ex1
    ./ex1 0
