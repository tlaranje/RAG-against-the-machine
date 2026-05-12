from tqdm import tqdm
from typing import Any


def bar(
    data: Any = "",
    desc: str = "Hello, World!",
    color: str = "white",
    total: int | None = None,
    position: int = 0,
    leave: bool = False
) -> tqdm:
    if total is None:
        try:
            total = len(data)
        except TypeError:
            total = None

    return tqdm(
        data,
        total=total,
        desc=desc,
        colour=color,
        position=position,
        leave=leave
    )
