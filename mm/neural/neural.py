from torch import Tensor
from mm.common import not_null


class Sample:
    values: list[str]
    probs: list[float]
    prob: float


class UpdateNetResult:
    learningRate: float;


def updateNet(parameters: list[Tensor],
              iteration: int,
              maxIteration: int,
              startLr: float,
              endLr: float) -> UpdateNetResult:
    res = UpdateNetResult()
    res.learningRate = max(endLr, startLr - (startLr - endLr) * iteration / maxIteration)
    for p in parameters:
       p.data += -res.learningRate * not_null(p.grad)
    return res


def calcOneProb(probs: list[float]) -> float:
    total = probs[0]
    length = len(probs)
    for p in probs[1:]:
        total *= p ** (1 / length)
    return total
