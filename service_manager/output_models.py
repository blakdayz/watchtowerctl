import subprocess

from typing import Any

from pydantic import BaseModel


class ServiceMetadata(BaseModel):
    label: str
    path: str
    dyld_dependencies: List[str] = []


class InvestigationArtifact(BaseModel):
    artifact_type: str  # e.g., "custom_instruction"
    data: Any  # The actual artifact data (e.g., instruction string for custom_instruction)


from typing import List, Union
from pydantic import BaseModel


class FDInfo(BaseModel):
    fd: int
    type: str
    device: Union[str, None] = None
    size: Union[int, None] = None
    node: Union[str, None] = None
    name: Union[str, None] = None


class NetworkConn(BaseModel):
    command: str
    pid: int
    user: str
    fd_info: List[FDInfo]
    address: str
    port: Union[str, None] = None
    state: str


class ProcessDetails(BaseModel):
    command: str
    pid: int
    user: str
    fds: List[FDInfo]


def parse_lsof_output(lines: list) -> List[Union[NetworkConn, ProcessDetails]]:
    result = []
    for line in lines:
        if "COMMAND" not in line and "PID" not in line:
            fields = line.split()
            if len(fields) >= 8:
                fd_info = FDInfo(
                    fd=int(fields[3]),
                    type=fields[4],
                    device=fields[5] if fields[5].isdigit() else None,
                    size=int(fields[6]) if fields[6].isdigit() else None,
                    node=fields[7],
                    name=fields[8],
                )

                # Check if the line contains network connection details
                if ":" in fields[-1] and "->" not in fields[-2]:
                    result.append(
                        NetworkConn(
                            command=fields[0],
                            pid=int(fields[1]),
                            user=fields[2],
                            fd_info=[fd_info],
                            address=fields[9].split(":")[0],
                            port=fields[9].split(":")[-1] if ":" in fields[9] else None,
                            state=fields[-1],
                        )
                    )
                else:
                    result.append(
                        ProcessDetails(
                            command=fields[0],
                            pid=int(fields[1]),
                            user=fields[2],
                            fds=[fd_info],
                        )
                    )

    return result


if __name__ == "__main__":

    output = subprocess.run("lsof -Pn -s -i", stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    parsed_result = parse_lsof_output(output[2:])  # Skip the header line

    for item in parsed_result:
        if isinstance(item, NetworkConn):
            print(
                f"Network Connection: {item.command} ({item.pid}) - {item.address}:{item.port} ({item.state})"
            )
        else:
            print(f"Process Detail: {item.command} ({item.pid}) - FDs: {item.fds}")
