from zope.container.contained import ContainedProxy


class ThreadLocalContainer:
    pass


class ThreadAcquisitionProxy(ContainedProxy):
    """Thread-level acquisition proxy mixin for use with Contained objects."""

    def __init__(self, wrapped):
        super().__init__(wrapped, ThreadLocalContainer)
        self.__thread_local = ThreadLocalContainer()

    def __getattr__(self, name):
        if name in ("__parent__", "__name__"):
            return getattr(self._wrapped, name)
        elif hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return getattr(self._wrapped, name)
            except AttributeError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )

    def __setattr__(self, name, value):
        """
        Sets
        :param name:
        :param value:
        :return:
        """
        if name in ("__parent__", "__name__"):
            setattr(self._wrappxed, name, value)
        elif hasattr(self, name) or not name.startswith("__"):
            super().__setattr__(name, value)
        else:
            setattr(self._wrapped, name, value)

    def __delattr__(self, name):
        if name in ("__parent__", "__name__"):
            delattr(self._wrapped, name)
        elif hasattr(self, name) or not name.startswith("__"):
            super().__delattr__(name)
        else:
            delattr(self._wrapped, name)

    def __dir__(self):
        return [attr for attr in super().__dir__() if not attr.startswith("__")] + list(
            self.__thread_local.keys()
        )

    def __getitem__(self, key):
        return self.__thread_local[key]

    def __setitem__(self, key, value):
        self.__thread_local[key] = value

    def __delitem__(self, key):
        del self.__thread_local[key]

    def has_key(self, key):
        return key in self.__thread_local
