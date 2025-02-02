import subprocess
import os
import shutil
import tempfile
import json
from typing import List


def get_network_connections(pid: int) -> List[NetworkConnection]:
    output = (
        subprocess.check_output(["lsof", "-Pn", "-s", "-i", "-p", str(pid)])
        .decode()
        .split("\n")
    )
    connections = []
    for line in output:
        if "COMMAND" not in line and line.strip():
            data = line.split()
            conn = {
                "pid": int(data[1]),
                "user": data[2],
                "address": data[4],
                "port": int(data[5].split(":")[1]) if ":" in data[5] else None,
                "state": data[7],
            }
            connections.append(conn)
    return [NetworkConnection(**conn) for conn in connections]


def get_process_info(pid: int) -> ProcessInfo:
    output = subprocess.check_output(
        ["lsof", "-Pn", "-s", "-i", "-p", str(pid)]
    ).decode()
    # ... (parse the output to create a ProcessInfo object)
