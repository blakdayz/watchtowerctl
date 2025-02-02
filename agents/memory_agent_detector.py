import os
import re
import asyncio
import zope.security.examples.sandbox
from typing import List, Dict, Any
import psutil
import yara

from agents.agent_base import Agent


class MemoryAgentDetectionError(Exception):
    """
    Exception raised when memory agent detection fails





















































    """


class MemoryInterviewAgent(Agent, zope.security.examples.sandbox.Agent):
    """
    Detects inject runtime agents in proccesses
    """

    def __init__(self, yara_rules_path: str):
        self.yara_rules = yara.compile(filepath=yara_rules_path)
        self.processes = psutil.process_iter()

    async def scan_memory(self, process: psutil.Process) -> List[Dict[str, Any]]:
        try:
            # Extract the memory dump as a byte array
            memory_dump = process.memory_info()[0]

            # Use Yara to search for matches in the memory dump
            yara_matches = self.yara_rules.match(data=memory_dump)

            network_listeners = await self._get_network_listeners(process)

            results = []
            for match in yara_matches:
                result = {
                    "pid": process.pid,
                    "label": match.string,
                    "score": match.confidence,
                    "offsets": [offset.address for offset in match.offsets],
                }
                results.append(result)

            # Add network listeners to the results
            results.extend(network_listeners)

            return results
        except Exception as e:
            print(f"Error scanning memory: {e}")
            raise MemoryAgentDetectionError("Error scanning memory") from e

    @staticmethod
    async def _get_network_listeners(process: psutil.Process) -> List[Dict[str, Any]]:
        network_listeners = []

        for proc in psutil.process_iter():
            try:
                # Get the process name
                process_name = proc.name()

                # Get the list of open file descriptors
                fd_list = proc.open_files()

                for fd in fd_list:
                    try:
                        # Check if the file descriptor is a network socket
                        if fd.path.endswith((".sock", ".so", ".dll")):
                            network_listeners.append(
                                {
                                    "process": process_name,
                                    "fd": fd.path,
                                    "type": "listener",
                                }
                            )

                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                        psutil.ZombieProcess,
                    ):
                        continue

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return network_listeners
