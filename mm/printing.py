from typing import Any
from datetime import datetime
import os


logFilePath = "log.txt"


def log(
    label: Any | None = "",
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n"
) -> None:
    lbl = "" if label == None else f"{(label + ':'):<20}"
    logSimple(lbl, *values, sep=sep, end=end)


def logSimple(
    label: Any | None = "",
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n"
) -> None:
    print(label, *values, sep=sep, end=end)
    with open(logFilePath, "a") as f:
        print(label, *values, sep=sep, end=end, file=f)


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
