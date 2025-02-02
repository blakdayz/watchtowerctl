import json

import os
import subprocess
import tempfile
from typing import List

from models.event_bus import Action
from models.procinfo import ProcessInfo


class SupervisingAgent:
    def __init__(self, virtual_env_path: str, containerization_platform: str):
        self.virtual_env_path = virtual_env_path
        self.containerization_platform = containerization_platform
        self.activated_processes: List[int] = []

    def activate_virtual_ENV(self) -> None:
        os.environ["VIRTUAL_ENV"] = self.virtual_env_path

    def run_socat_dump_comms(
        self, network_connections: List[NetworkConnection]
    ) -> None:
        for conn in network_connections:
            if conn.state == "ESTABLISHED":
                command = generate_socat_dump_command(
                    conn.address, conn.port, "dump.pcap"
                )
                subprocess.run(command.split(), check=True)

    def run_socat_sinkhole_comms(
        self,
        network_connections: List[NetworkConnection],
        sinkhole_address: str,
        sinkhole_port: int,
    ) -> None:
        for conn in network_connections:
            if conn.state == "ESTABLISHED":
                command = generate_socrat_sinkhole_command(
                    conn.address, conn.port, sinkhole_address, sinkhole_port
                )
                subprocess.run(command.split(), check=True)

    def gather_artifacts(
        self, process_info: ProcessInfo, network_connections: List[NetworkConnection]
    ) -> ArtifactInfo:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = os.path.join(temp_dir, "artifacts")
            os.makedirs(artifact_dir, exist_ok=True)

            # Gather relevant artifacts (process_info, network_connections, etc.)
            artifact_data = {
                "process_info": process_info.dict(),
                "network_connections": [conn.dict() for conn in network_connections],
                # Add other relevant artifact data here
            }

            with open(os.path.join(artifact_dir, "artifacts.json"), "w") as f:
                f.write(json.dumps(artifact_data))

            return ArtifactInfo(
                process_info=process_info, network_connections=network_connections
            )

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

    @classmethod
    def from_pid(
        cls, pid: int, virtual_env_path: str, containerization_platform: str
    ) -> "Agent":
        process_info = get_process_info(pid)
        network_connections = get_network_connections(pid)
        return cls(
            virtual_env_path,
            containerization_platform,
            process_info,
            network_connections,
        )

    def __enter__(self) -> "Agent":
        self.activate_virtual_ENV()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Perform cleanup tasks here
        pass

    def generate_action(process_info: ProcessInfo) -> Action:
        if process_info.threat_level == ThreatLevel.NONE or process_info.pid is None:
            return (
                Action.DISABLE_SERVICE
            )  # Default action when no threat is detected or PID is unknown

        elif process_info.command in ["launchctl", "bootout"]:
            return Action.BOOTOUT  # Stop the launch daemon or bootout process directly

        elif any(
            conn.state == "ESTABLISHED" for conn in process_info.network_connections
        ):
            if process_info.pid:
                return Action.KILL_PROCESS | Action.SOCAT_DUMP
            else:
                return (
                    Action.SOCAT_DUMP
                )  # Dump network traffic without killing the process

        elif any(conn.command == "soc" for conn in process_info.network_connections):
            return (
                Action.SOCAT_SINKHOLE
            )  # Sinkhole socat connections to stop all communications

        else:
            if process_info.pid is not None:
                return Action.KILL_PROCESS | Action.LAUNCHCTL_REMOVE
            else:
                return (
                    Action.LAUNCHCTL_REMOVE
                )  # Remove the offending launch daemon (if applicable) without killing any processes
