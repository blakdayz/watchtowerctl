import json
import os
import subprocess
import tempfile
from typing import List

from agents.agent_interface import Jinja2CommandBuilder, AgentBase
from models.procinfo import ProcessInfo


class Agent(AgentBase):
    def __init__(self, virtual_env_path: str, containerization_platform: str, jinja2_command_builder: Jinja2CommandBuilder = None):
        self.virtual_env_path = virtual_env_path
        self.containerization_platform = containerization_platform
        self.jinja2_command_builder = jinja2_command_builder or Jinja2CommandBuilder()
        self.activated_processes: List[int] = []

    def activate_virtual_ev(self) -> None:
        os.environ["VIRTUAL_ENV"] = self.virtual_env_path

    def run_socat_dump_comms(self, network_connections: List[NetworkConnection]) -> None:
        for conn in network_connections:
            if conn.state == "ESTABLISHED":
                command = self.jinja2_command_builder.generate_socat_dump_command(conn.address, conn.port, "dump.pcap")
                subprocess.run(command.split(), check=True)

    def run_socat_sinkhole_comms(self, network_connections: List[NetworkConnection], sinkhole_address: str, sinkhole_port: int) -> None:
        for conn in network_connections:
            if conn.state == "ESTABLISHED":
                command = self.jinja2_command_builder.generate_socat_sinkhole_command(conn.address, conn.port, sinkhole_address, sinkhole_port)
                subprocess.run(command.split(), check=True)

    def gather_artifacts(self, process_info: ProcessInfo, network_connections: List[NetworkConnection]) -> ArtifactInfo:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = os.path.join(temp_dir, "artifacts")
            os.makedirs(artifact_dir, exist_ok=True)

            # Gather relevant artifacts (process_info, network_connections, etc.)
            artifact_data = {
                "process_info": process_info.dict(),
                "network_connections": [conn.dict() for conn in network_connections],
                "binary_target": []
            }

            with open(os.path.join(artifact_dir, "artifacts.json"), "w") as f:
                f.write(json.dumps(artifact_data))

            return ArtifactInfo(process_info=process_info, network_connections=network_connections)

    def inspect_drivers(self) -> None:
        # Implement driver inspection logic here
        pass

    def unload_remove_bootout(self, pid: int) -> None:
        if self.containerization_platform == "docker":
            subprocess.run(["docker", "kill", str(pid)])
        elif self.containerization_platform == "podman":
            subprocess.run(["podman", "kill", str(pid)])

    def activate_process(self, process_info: ProcessInfo) -> None:
        if process_info.pid not in self.activated_processes:
            self.activated_processes.append(process_info.pid)
            # Implement activation logic for the given process here

    def deactivate_process(self, pid: int) -> None:
        if pid in self.activated_processes:
            self.activated_processes.remove(pid)
