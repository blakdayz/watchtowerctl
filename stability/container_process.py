from .base_thread_container import monitor_object_access
from agents.memory_agent_detector import MemoryAgentDetector
from sample_container import SampleContainer

class ContainerizedProcess:
    def __init__(self):
        self.container = SampleContainer()
        self.detector = MemoryAgentDetector("path/to/your/yara_rules.yar")
        self.container['detector'] = self.detector

import weakref, ThreadNetwork, Network

class ContainerizedThreadNetwork(ThreadNetwork.ThreadNetwork):
    def __init__(self):
        ThreadNetwork.__init__(self)
        self.container = SampleContainer()
        self.container['detector'] = self.detector
        self.container['network'] = Network()
        self.container['container'] = weakref.proxy(self)



class ContainerController:
    """
    A manager that enforces, this is the
    """
    def __init__(self, container):
        self.container = weakref.ref(container)
        self.controlled_objects = set()

    def _wrap_object(self, obj):
        """Wrap an object with a proxy that monitors `__get__` and `__set__` attributes."""
        return monitor_object_access(obj)

    def _unwrap_object(self, obj):
        """Unwrap an object from the observer pattern if it was previously wrapped."""
        if hasattr(obj, '_wrapped'):
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
