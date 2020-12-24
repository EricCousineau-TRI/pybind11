# try:
#     import enum
# except ModuleNotFoundError:
#     enum = None

# if enum is not None:
if False:

    def _instancecheck(cls, instance):
        print(("instance", cls, instance))
        subclass = type(instance)
        return issubclass(subclass, cls)

    def _subclasscheck(cls, subclass):
        print(("subclass", cls, subclass))
        if type.__subclasscheck__(enum.Enum, cls) and type.__subclasscheck__(cls, subclass):
            return True
        elif type.__subclasscheck__(pybind11_base_cls, cls):
            is_pybind11_enum = getattr(subclass, '_is_pybind11_enum')
            return is_pybind11_enum == True
        else:
            return False

    enum.EnumMeta.__instancecheck__ = _instancecheck
    enum.EnumMeta.__subclasscheck__ = _subclasscheck


class pybind11_enum_meta_cls(pybind11_meta_cls):
    def __iter__(cls):
        return iter(cls.__members__.values())

    def __len__(cls):
        return len(cls.__members__)


pybind11_enum_base_cls = None
