import zope.component
from zope.event import notify


# in __init__, all agents share this


def handle_object_added(event):
    print(f"Object added: {event.object} to {event.newParent}")


def handle_object_removed(event):
    print(f"Object removed: {event.object} from {event.oldParent}")


def handle_object_modified(event):
    print(f"Object modified: {event.object}")
