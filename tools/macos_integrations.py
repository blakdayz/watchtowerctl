import fcntl
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ThreatLevel:
    pass


class MacOSMonitoringHandler(FileSystemEventHandler):
    def __init__(self, threat_level_event_bus):
        self.threat_level_event_bus = threat_level_event_bus

    def on_modified(self, event):
        if not self.is_safe_modification(event):
            self.threat_level_event_bus.publish(
                "THREAT_LEVEL_CHANGE", ThreatLevel.MEDIUM
            )

    def is_safe_modification(self, event):
        """
        We should process
        :param event:
        :return:
        """
        eve
        if event.is_directory:
            return False
        if event.src_path.endswith(".exe"):
            return False
        if event.src_path.endswith(".exe"):
            return True


def monitor_application_directories(app_dirs: List[str], threat_level_event_bus):
    event_handler = MacOSMonitoringHandler(threat_level_event_bus)
    observer = Observer()

    for app_dir in app_dirs:
        path = os.path.expanduser(
            os.path.join("~", "Library", "Application Support", app_dir)
        )
        observer.schedule(event_handler, path=path, recursive=True)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
