from typing import List, Union, Dict, Any
from pydantic import BaseModel

class InfoDict(BaseModel):
    key: str
    value: Union[str, int, float]

class PathInfo(BaseModel):
    path: str
    type: Union[str, None] = None
    mode: Union[int, None] = None
    dev_major: Union[int, None] = None
    dev_minor: Union[int, None] = None

class FDInfo(BaseModel):
    fd: int
    type: str
    path_info: PathInfo

class ProcessInfo(BaseModel):
    pid: int
    ppid: int
    pgid: int
    sid: int
    command: str
    args: List[str]
    env: Dict[str, str]
    cwd: Union[PathInfo, None] = None
    root: Union[PathInfo, None] = None
    fds: List[FDInfo] = []
    memory_maps: List[Dict[str, Any]] = []
    environment_variables: Dict[str, str] = {}
    status: str
    start_time: int
    cpu_usage: float
    memory_size: int
    threads: int
    info_dicts: List[InfoDict] = []

def parse_procinfo_output(lines: list) -> ProcessInfo:
    """
    Run p
    The output of running launchctl procinfo <pid>

    :param lines:
    :return:
    """
    result = ProcessInfo(pid=int(lines[1].split()[0]))

    # Extract basic process details
    for line in lines[2:-5]:
        if ":" in line:
            key, value = line.split(":", 1)
            setattr(result, key.lower().replace(" ", "_"), value.strip())

    # Extract environment variables and memory maps
    env_lines = lines[-4].splitlines()
    memmap_lines = lines[-3:-2][0].splitlines()

    result.environment_variables = {line.split("=", 1)[0]: line.split("=", 1)[1] for line in env_lines if "=" in line}
    result.memory_maps = [dict(line.split(maxsplit=1)) for line in memmap_lines]

    # Extract FD info and path info
    fds_lines = lines[-2].splitlines()
    for fd_line in fds_lines:
        fd_info = FDInfo(fd=int(fd_line.split()[0]), type=" ".join(fd_line.split()[1:-4]))
        path_info = PathInfo(**{k: v for k, v in zip("path mode dev_major dev_minor".split(), fd_line.split())[4:]})

        result.fds.append(FDInfo(fd=fd_info.fd, type=fd_info.type, path_info=path_info))

    # Extract info_dicts
    result.info_dicts = [InfoDict(key=k, value=v) for k, v in zip(*map(lambda x: x.split(":", 1), lines[-5].split()))]

    return result
