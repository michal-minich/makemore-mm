from typing import Optional, TypeVar



def findLowestIndex(arr: list) -> int:
    ix = 0
    for i in range(1, len(arr)):
        if arr[i] < arr[ix]:
            ix = i
    return ix


def readFileSplitByLine(name: str) -> list[str]:
    words = open(name, "r", encoding="utf-8").read().splitlines()
    return words


T = TypeVar('T')


def not_null(value: Optional[T]) -> T:
    if value is not None:
        return value
    else:
        raise ValueError("Value cannot be None.")
