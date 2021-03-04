#!/bin/bash
set -eux

# sudo apt install python3.8-dev python3.8-venv python3.8-dbg

python3.8-dbg -m venv ./venv
source ./venv/bin/activate
pip install -U pip wheel
pip install pytest

mkdir build && cd build
cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DPYBIND11_TEST_OVERRIDE=test_multiple_inheritance.cpp

env PYTHONUNBUFFERED=1 PYTEST_ADDOPTS="-s -x" ninja pytest

src_dir=${PWD}/..
build_dir=${PWD}
( cd ${build_dir}/tests && gdb --args python -m pytest ${src_dir}/tests/test_multiple_inheritance.py )
