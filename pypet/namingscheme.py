"""Module that add prefix functionality"""


__author__ = 'Robert Meyer'


from pypet.slots import MetaSlotMachine, add_metaclass
from pypet.pypetlogging import HasLogger


class MetaNamingScheme(MetaSlotMachine):
    """Meta-class that adds the `f_` and `v_` naming scheme to a class.

    `__api__`  and `__prefix_api__` is a set that contains all functions and
    property names, where everything in `__prefix_api__` starts with
    the prefix `f_` or `v_`.

    """
    def __init__(cls, name, bases, dictionary):
        super(MetaNamingScheme, cls).__init__(name, bases, dictionary)
        all_elements = dir(cls)
        api = set()
        new_api = set()
        for name in all_elements:
            if not name.startswith('_'):
                item = getattr(cls, name)
                if hasattr(item, '__call__'):
                    if name.startswith('f_'):
                        new_name = name
                    else:
                        api.add(name)
                        new_name = 'f_' + name
                        setattr(cls, new_name, item)
                    new_api.add(new_name)
                elif isinstance(item, property):
                    if name.startswith('v_'):
                        new_name = name
                    else:
                        new_name = 'v_' + name
                        api.add(name)
                        setattr(cls, new_name, item)
                    new_api.add(new_name)

        cls.__api__ = api
        cls.__prefix_api__ = new_api


@add_metaclass(MetaNamingScheme)
class PypetNaming(HasLogger):
    """Abstract class that allows normal and prefix API"""
    __slots__ = ()

    def __dir__(self):
        """Returns only elements that are part of the prefix API"""
        dir_list = super(PypetNaming, self).__dir__()
        dir_list = [x for x in dir_list if (x in self.__prefix_api__ or x not in self.__api__)]
        return dir_list

    def __all_dir__(self):
        """Returns all elements regardless of API"""
        return super(PypetNaming, self).__dir__()

    def __getattr__(self, item):
        if item.startswith('v_'):
            short = item[2:]
            if short in self.__all_slots__:
                return getattr(self, short)
            try:
                return self.__dict__[short]
            except KeyError:
                pass
        raise AttributeError('`%s` objct has not attribute '
                                 '`%s`' % (self.__class__.__name__, item))

# class Test(PypetNamingScheme):
#
#     def __init__(self):
#         self._v = 42
#     def f(self):
#         return 42
#
#     @property
#     def v(self):
#         return self._v
#
#     @v.setter
#     def v(self, val):
#         self._v = val
#
#     @property
#     def g(sell):
#         return 9
#
#
# y = Test()
# dir(y)
# y.v_v = 3
# y.v = 19
# y.g = 22
# pass
