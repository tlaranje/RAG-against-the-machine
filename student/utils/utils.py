from tqdm import tqdm
from typing import Any


def bar(
    data: Any = "",
    desc: str = "Hello, World!",
    color: str = "white",
    total: int = 100,
    position: int = 0,
    leave: bool = False
) -> tqdm:
    return tqdm(
        data,
        total=total,
        desc=desc,
        colour=color,
        position=position,
        leave=leave
    )
