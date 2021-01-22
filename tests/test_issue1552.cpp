// https://github.com/pybind/pybind11/issues/1552
#include <iostream>

#include "pybind11_tests.h"

class Dispatcher;

class Client
{
public:
    Client(Dispatcher* disp): PtrD(disp)
    {
        std::cout << "In Client::Client\n";
    }
    virtual ~Client(){};
    virtual void ProcessEvent()
    {
        std::cout << "THIS SHOULDN'T HAPPEN --In Client::ProcessEvent\n";
    }
    Dispatcher* PtrD;
};

class Dispatcher
{
public:
    Dispatcher()
    {
        std::cout << "In Dispatcher::Dispatcher\n";
    }
    virtual ~Dispatcher(){};

    void Dispatch(Client* client)
    {
        std::cout << "Dispatcher::Dispatch called by " << client << std::endl;
        client->ProcessEvent();
    }
};

class DispatcherTrampoline : public Dispatcher
{
public:
    using Dispatcher::Dispatcher;
};

class ClientTrampoline : public Client
{
public:
    using Client::Client;

    void ProcessEvent() override
    {
        PYBIND11_OVERLOAD(void,Client,ProcessEvent,);
    }
};

TEST_SUBMODULE(issue1552, m)
{
    py::class_<Client,ClientTrampoline> cli(m,"Client");
    cli.def(py::init<Dispatcher* >());
    cli.def("ProcessEvent",&Client::ProcessEvent);
    cli.def_readwrite("PtrD",&Client::PtrD);

    py::class_<Dispatcher,DispatcherTrampoline> dsp(m,"Dispatcher");
    dsp.def(py::init< >());
    dsp.def("Dispatch",&Dispatcher::Dispatch);
}
