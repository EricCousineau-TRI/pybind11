// https://github.com/pybind/pybind11/issues/1552
#include <iostream>

#include "pybind11_tests.h"

class Dispatcher;

class Client
{
public:
    Client()
    {
        std::cout << "In Client::Client\n";
    }
    virtual ~Client(){};
    virtual void ProcessEvent() = 0;
};

class Dispatcher
{
public:
    Dispatcher()
    {
        std::cout << "In Dispatcher::Dispatcher\n";
    }

    void Dispatch(Client* client)
    {
        std::cout << "Dispatcher::Dispatch called by " << client << std::endl;
        client->ProcessEvent();
    }
};

class ClientTrampoline : public Client
{
public:
    using Client::Client;

    void ProcessEvent() override
    {
        PYBIND11_OVERLOAD_PURE(void,Client,ProcessEvent,);
    }
};

TEST_SUBMODULE(issue1552, m)
{
    py::class_<Client,ClientTrampoline> cli(m,"Client");
    cli.def(py::init());
    cli.def("ProcessEvent",&Client::ProcessEvent);

    py::class_<Dispatcher> dsp(m,"Dispatcher");
    dsp.def(py::init< >());
    dsp.def("Dispatch",&Dispatcher::Dispatch);
}
