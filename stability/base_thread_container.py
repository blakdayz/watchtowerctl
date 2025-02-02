import types

from stability.object_control_graph import SampleContainer


def _monitored_get__(self):
    # Log access to the object and return the wrapped object's value
    print(f"Accessing object {id(self._wrapped)} ({type(self._wrapped).__name__})")
    return self._get_wrapped().__get__(self)


def _monitored_set__(self, value):
    # Log setting of the object's attribute and set the wrapped object's attribute accordingly
    print(
        f"Setting object {id(self._wrapped)} ({type(self._wrapped).__name__}) to {value}"
    )
    self._get_wrapped().__set__(self, value)


def monitor_object_access(obj):
    """Wrap an object with a proxy that monitors `__get__` and `__set__` attributes."""
    if isinstance(obj, types.BuiltinFunctionType) or not hasattr(obj, "__get__"):
        return obj

    wrapped_obj = obj._wrapped if hasattr(obj, "_wrapped") else obj
    new_proxy = type(
        obj.__class__.__name__,
        (),
        {
            **vars(obj),
            "__get__": _monitored_get__,
            "__set__": _monitored_set__,
            "_wrapped": wrapped_obj,
            "_get_wrapped": getattr(wrapped_obj, "__get__", lambda: obj),
        },
    )

    return new_proxy()


import weakref


class ContainerController:
    def __init__(self, container):
        self.container = weakref.ref(container)
        self.controlled_objects = set()

    def _wrap_object(self, obj):
        """Wrap an object with a proxy that monitors `__get__` and `__set__` attributes."""
        return monitor_object_access(obj)

    def _unwrap_object(self, obj):
        """Unwrap an object from the observer pattern if it was previously wrapped."""
        if hasattr(obj, "_wrapped"):
            return obj._wrapped
        return obj

    def control_objects(self, *objects):
        """
        Control a list of objects by wrapping them with the observer pattern.

        This method will wrap each object with a proxy that monitors `__get__` and `__set__` attributes,
        allowing you to control and log object behavior at runtime.
        """
        container = self.container()
        if not container:
            return

        for obj in objects:
            key = id(obj)
            if key in self.controlled_objects:
                continue

            wrapped_obj = self._wrap_object(obj)
            setattr(container, str(key), wrapped_obj)
            self.controlled_objects.add(key)

    def release_control(self, *objects):
        """
        Release control of a list of objects by unwrapping them from the observer pattern.

        This method will unwrap each object from the observer pattern and restore their original behavior.
        """
        container = self.container()
        if not container:
            return

        for obj in objects:
            key = id(obj)
            if key not in self.controlled_objects:
                continue

            unwrapped_obj = self._unwrap_object(getattr(container, key))
            setattr(container, str(key), unwrapped_obj)
            delattr(container, str(key))
            self.controlled_objects.remove(key)


def main():
    container = SampleContainer()
    controller = ContainerController(container)

    # Control some objects in the container
    obj1 = object()
    obj2 = "hello"
    controller.control_objects(obj1, obj2)
    print(f"{container[1]} {container['obj2']}")

    # Access and modify a controlled object
    container[1].some_attribute = 42
    print(f"{container[1].__dict__}")

    # Release control of the objects
    controller.release_control(obj1, obj2)
    print(f"{container[1]} {container['obj2']}")


if __name__ == "__main__":
    main()
