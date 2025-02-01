import os
import sys
import types
import asyncio
from enum import Enum
from typing import List, Dict, Any

from zope.interface import classImplements
from zope.interface import implementedBy

from stability.thread_acquisition_proxy import ThreadAcquisitionProxy
from stability.base_thread_container import ContainerController



from zope.interface import implementer


from thread_acquisition_proxy import ThreadAcquisitionProxy

class ContainmentStatus(Enum):
    CONTAINED = 1
    UNCONTAINED = 2
    IN_CONTROL = 3
    GB_COLLECTED =4

class IContainer:
    status: ContainmentStatus # = ContainmentStatus.CONTAINED


@implementer(IContainer)
class SampleContainer(IContainer, metaclass=ThreadAcquisitionProxy):
    """Sample container implementation suitable for testing.

    It is not suitable, directly as a base class unless the subclass
    overrides `_newContainerData` to return a persistent mapping object.
    """

    def __init__(self):
        super().__init__()
        self.__data = self._newContainerData()

    # ... (other methods remain mostly unchanged)

    def _newContainerData(self):
        """Construct an item-data container

        Subclasses should override this if they want different data.

        The value returned is a mapping object that also has `get`,
        `has_key`, `keys`, `items`, and `values` methods.
        """
        return ThreadAcquisitionProxy(ThreadLocalContainer())


class ObjectBehaviorEvent:
    def __init__(self, obj: Any):
        self.obj = obj

    def get_object(self) -> Any:
        return self.obj


class ObjectStructureChangedEvent(ObjectBehaviorEvent):
    pass



class ObjectControlManager:

    def __init__(self, container: SampleContainer):
        self.container = container
        self.controller = ContainerController(container)
        self.graph = {}

        # Initialize the object control graph by monitoring all objects in the container
        self._monitor_objects(self.container.objects())

    async def _monitor_object(self, obj: Any) -> None:
        wrapped_obj = ThreadAcquisitionProxy(obj)

        # Monitor object behavior events (e.g., attribute access and modification)
        for attr_name in dir(wrapped_obj):
            if not attr_name.startswith('__'):
                continue

            attr = getattr(wrapped_obj, attr_name)
            if callable(attr):

                # Wrap the attribute with a proxy that emits an event when accessed
                async_attr = self._wrap_attribute_with_event_emitter(obj, attr)

                setattr(wrapped_obj, attr_name, async_attr)

        # Monitor object structural changes (e.g., attribute addition and removal)
        for attr_name in dir(obj):
            if not attr_name.startswith('__'):
                continue

            attr = getattr(obj, attr_name)
            if callable(attr):

                # Add an event listener to the attribute that emits a structural change event when accessed
                self._add_structural_change_event_listener(obj, attr_name)

        # Add the object to the control graph
        self._add_object_to_graph(obj)

    def _monitor_objects(self, objects: List[Any]) -> None:
        for obj in objects:
            asyncio.create_task(self._monitor_object(obj))

    def _wrap_attribute_with_event_emitter(self, obj: Any, attr: Any) -> Any:
        """Wrap an attribute with a proxy that emits an event when accessed."""
        if not callable(attr):
            return attr

        async def access_and_emit_event(*args, **kwargs):
            event = ObjectBehaviorEvent(obj)
            await self._emit_object_behavior_event(event)

            result = attr(*args, **kwargs)
            return result

        return types.FunctionType(
            access_and_emit_event,
            obj.__class__,
            attr.__name__,
            args=attr.__,
            closure=attr.__closure__
        )

    def _add_structural_change_event_listener(self, obj: Any, attr_name: str) -> None:
        """Add an event listener to an attribute that emits a structural change event when accessed."""
        if not callable(getattr(obj, attr_name)):
            return

        def structural_change_event_listener(*args, **kwargs):
            event = ObjectStructureChangedEvent(obj)
            self._emit_object_structural_change_event(event)

        setattr(
            obj,
            attr_name,
            types.MethodType(structural_change_event_listener, obj.__class__, attr_name),
        )

    def _add_object_to_graph(self, obj: Any) -> None:
        """Add an object to the control graph."""
        if obj in self.graph:
            return

        # Add the object's ID and structural properties (e.g., binary location on disk) to the graph
        self._add_object_id_and_structural_properties(obj)

        # Add the object's behavior events (e.g., attribute access and modification) to the graph
        self._add_object_behavior_events_to_graph(obj)

    def _add_object_id_and_structural_properties(self, obj: Any) -> None:
        """Add an object's ID and structural properties (e.g., binary location on disk) to the control graph."""
        if not hasattr(obj, 'id'):
            setattr(obj, 'id', id(obj))

        # Add other structural properties as needed
        self._add_object_structural_property(obj, 'binary_location_on_disk', obj.__file__)

    def _add_object_behavior_events_to_graph(self, obj: Any) -> None:
        """Add an object's behavior events (e.g., attribute access and modification) to the control graph."""
        # Add event listeners for object behavior events
        for attr_name in dir(obj):
            if not attr_name.startswith('__'):
                continue

            attr = getattr(obj, attr_name)
            if callable(attr):

                self._add_object_behavior_event_listener(obj, attr_name)

    @staticmethod
    def _add_object_structural_property(obj: Any, prop_name: str, value: Any) -> None:
        """Add an object's structural property (e.g., binary location on disk) to the control graph."""
        setattr(obj, prop_name, value)

    def _add_object_behavior_event_listener(self, obj: Any, event_name: str) -> None:
        """Add an event listener for an object's behavior event (e.g., attribute access and modification)."""
        if not callable(getattr(obj, event_name)):
            return

        def behavior_event_listener(*args, **kwargs):
            event = ObjectBehaviorEvent(obj)
            self._emit_object_behavior_event(event)

        setattr(
            obj,
            event_name,
            types.MethodType(behavior_event_listener, obj.__class__, event_name),
        )

    async def _emit_object_behavior_event(self, event: ObjectBehaviorEvent) -> None:
        """Emit an object behavior event."""
        # Add the event to the control graph
        self._add_graph_edge(event.obj.id, 'behaves_like', event.obj)

    async def _emit_object_structural_change_event(self, event: ObjectStructureChangedEvent) -> None:
        """Emit an object structural change event."""
        # Add the event to the control graph
        self._add_graph_edge(event.obj.id, 'changed_structure_of', event.obj)

    def _add_graph_edge(self, node_id: Any, edge_type: str, neighbor_id: Any) -> None:
        """Add an edge between two nodes in the control graph."""
        if node_id not in self.graph:
            self.graph[node_id] = set()

        self.graph[node_id].add((neighbor_id, edge_type))

    def control_objects(self, *objects: Any) -> None:
        """
        Control a list of objects by wrapping them with the observer pattern using the container controller.

        This method will wrap each object with a proxy that monitors `__get__` and `__set__` attributes,
        allowing you to control and log object behavior at runtime.
        """
        self.controller.control_objects(*objects)

    def release_control(self, *objects: Any) -> None:
        """
        Release control of a list of objects by unwrapping them from the observer pattern using the container controller.

        This method will unwrap each object from the observer pattern and restore their original behavior.
        """
        self.controller.release_control(*objects)
