# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import issue1552 as m


class SomeClient(m.Client):
    def __init__(self):
        print("In SomeClient::__init__")
        super().__init__()

    def ProcessEvent(self):
        print("Python ProcessEvent")


def test_main():
    # https://github.com/pybind/pybind11/issues/1552
    dd = m.Dispatcher()
    cl = SomeClient()
    dd.Dispatch(cl)
