# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import issue1552 as m


class SomeClient(m.Client):
    def __init__(self,d):
        print("In SomeClient::__init__")
        super().__init__(d);

    def ProcessEvent(self):
        print("in SomeClient::ProcessEvent,about to call self.ProcessEvent")
        self.PtrD.Dispatch(self);


def test_main():
    # https://github.com/pybind/pybind11/issues/1552
    dd = m.Dispatcher()
    cl = SomeClient(dd)
    dd.Dispatch(cl)
