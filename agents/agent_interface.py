import subprocess
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import List

from jinja2 import Environment, FileSystemLoader
from .models import ProcessInfo, FDInfo, PathInfo, NetworkConnection, ArtifactInfo

class AgentBase(ABC):
    @abstractmethod
    def unload_remove_bootout(self, pid: int) -> None:
        pass

    @abstractmethod
    def run_socat_dump_comms(self, network_connections: List[NetworkConnection]) -> None:
        pass

    @abstractmethod
    def run_socat_sinkhole_comms(self, network_connections: List[NetworkConnection]) -> None:
        pass

    @abstractmethod
    def gather_artifacts(self, process_info: ProcessInfo, network_connections: List[NetworkConnection]) -> ArtifactInfo:
        pass

    @abstractmethod
    def inspect_drivers(self) -> None:
        pass
