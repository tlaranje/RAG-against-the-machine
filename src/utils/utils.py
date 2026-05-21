from typing import Any
from tqdm import tqdm


def bar(
    data: Any = "",
    desc: str = "Hello, World!",
    color: str = "white",
    total: int | None = None,
    position: int = 0,
    leave: bool = False
) -> tqdm:
    """
    Creates a configured tqdm progress bar with custom styling.

    Args:
        data: Any iterable object to track progress over.
        desc: A string prefix displayed before the progress bar.
        color: Terminal color name string for the bar graphics.
        total: Expected number of total iterations.
        position: Specifier for the vertical terminal alignment index.
        leave: If True, keeps the completed progress bar visual intact.

    Returns:
        An active tqdm progress bar object wrapper.
    """
    # Safe guard: If total is not explicitly provided, try to automatically
    # calculate the length of the data structure using len().
    if total is None:
        try:
            total = len(data)
        except TypeError:
            # If the data type does not support len() (like generator streams),
            # fall back to None to render an infinite streaming indicator.
            total = None

    return tqdm(
        data,
        total=total,
        desc=desc,
        colour=color,
        position=position,
        leave=leave
    )
