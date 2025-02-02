from typing import Optional

from pydantic import BaseModel


class ProcessInfo(BaseModel):
    command: str
    pid: int
    user: str
    fd: int
    type: str
    device: Optional[str] = None
    size: Optional[int] = None
    node: Optional[int] = None
    name: Optional[str] = None

    class Config:
        orm_mode = True


class LsofOutput(BaseModel):
    entries: list[ProcessInfo]

    class Config:
        orm_mode = True
