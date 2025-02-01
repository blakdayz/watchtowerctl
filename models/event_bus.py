from collections import defaultdict
from enum import Enum

class EventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)

    def subscribe(self, event_type: str, callback):
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, *args, **kwargs):
        for callback in self.subscribers.get(event_type, []):
            callback(*args, **kwargs)

class EventType(str, Enum):
    PROCESS_INFO_UPDATED = "PROCESS_INFO_UPDATED"
    THREAT_LEVEL_CHANGE = "THREAT_LEVEL_CHANGE"
    ACTION_EXECUTION_REQUESTED = "ACTION_EXECUTION_REQUESTED"
    ACTION_EXECUTION_OUTPUT = "ACTION_EXECUTION_OUTPUT"


class Action(Enum):
    LAUNCHCTL_REMOVE = "LAUNCHCTL_REMOVE"
    BOOTOUT = "BOOTOUT"
    DISABLE_SERVICE = "DISABLE_SERVICE"
    KILL_PROCESS = "KILL_PROCESS"
    SOCAT_DUMP = "SOCAT_DUMP"
    SOCAT_SINKHOLE = "SOCAT_SINKHOLE"



