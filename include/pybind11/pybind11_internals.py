class pybind11_enum_meta_cls(pybind11_meta_cls):
    is_pybind11_enum = True

    def __iter__(cls):
        return iter(cls.__members__.values())

    def __len__(cls):
        return len(cls.__members__)


pybind11_enum_base_cls = None
