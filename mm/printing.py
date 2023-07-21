from typing import Any
from datetime import datetime
import os
from torch import Size


logFilePath = "logs/log.txt"


def log(
    label: Any | None = "",
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n"
) -> None:
    lbl = "" if label == None else f"{(label + ':'):<20}"
    logSimple(lbl, *values, sep=sep, end=end)


def logSimple(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n"
) -> None:
    vs = [getSizeString(v) if isinstance(v, Size) else v for v in values]
    print(*vs, sep=sep, end=end)
    with open(logFilePath, "a") as f:
        print(*vs, sep=sep, end=end, file=f)


def logSection(title: str) -> None:
    log(title, "-------------------------- " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def initLogging(title: str) -> None:
    currentDateTime = datetime.now()
    currentDateTimeStr = currentDateTime.strftime("%Y-%m-%d_%H_%M_%S")
    logsPath = "./logs/"
    global logFilePath 
    logFilePath = logsPath + currentDateTimeStr + ".txt"
    if not os.path.exists(logsPath):
        os.makedirs(logsPath)
    logSection(title)


def getSizeString(size: Size) -> str: 
    return "[" + ", ".join(str(dim) for dim in size) + "]"
