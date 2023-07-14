import torch
import torch.nn.functional as F
from torch import Tensor
from mm.printing import *
from mm.common import *
from mm.neural.layers import *
from mm.neural.network import *
from mm.neural.dataSet import *


class CalibrationResult:
    mean: Tensor
    std: Tensor


@torch.no_grad()
def calibrateBatchNorm(np: NetParameters, trX: Tensor) -> CalibrationResult:
    r = CalibrationResult()
    emb = np.C[trX]
    embCat = emb.view(emb.shape[0], -1)
    hPreActivations = embCat @ np.W1 #+ np.b1
    r.mean = hPreActivations.mean(0, keepdim=True)
    r.std  = hPreActivations.std(0, keepdim=True)
    return r


class Loss:
    hPreActivations: Tensor
    h: Tensor
    logits: Tensor
    loss: Tensor


class Loss2:
    logits: Tensor
    loss: Tensor


def getLoss(np: NetParameters,
            cal: CalibrationResult,
            emb: Tensor,
            y: Tensor) -> Loss:
    r = Loss()
    embCat = emb.view(emb.shape[0], -1)
    r.hPreActivations = embCat @ np.W1 #+ np.b1
    r.hPreActivations = np.batchNormGain * (r.hPreActivations - cal.mean) / cal.std + np.batchNormBias
    r.h = torch.tanh(r.hPreActivations)
    r.logits = r.h @ np.W2 + np.b2
    r.loss = F.cross_entropy(r.logits, y)
    return r


def getLoss2(np: NetParameters2,
             emb: Tensor,
             y: Tensor) -> Loss2:
    r = Loss2()
    r.logits = getLogits2(np, emb)
    r.loss = F.cross_entropy(r.logits, y)#.long())
    return r


def getLogits2(np: NetParameters2, emb: Tensor) -> Tensor:
    logits = emb.view(emb.shape[0], -1)
    for l in np.layers:
        logits = l(logits)
    return logits


class ForwardPassResult(Loss):
    emb: Tensor


def forwardPass(np: NetParameters,
                cal: CalibrationResult,
                trX: Tensor,
                trY: Tensor,                
                miniBatchIxs: Tensor) -> ForwardPassResult:
    r = ForwardPassResult()
    trBatchX = trX[miniBatchIxs]
    trBatchY = trY[miniBatchIxs]
    r.emb = np.C[trBatchX]
    loss = getLoss(np, cal, r.emb, trBatchY)
    r.hPreActivations = loss.hPreActivations
    r.h = loss.h
    r.logits = loss.logits
    r.loss = loss.loss
    return r


def forwardPass2(np: NetParameters2,
                 trX: Tensor,
                 trY: Tensor,                
                 miniBatchIxs: Tensor) -> ForwardPassResult:
    r = ForwardPassResult()
    trBatchX = trX[miniBatchIxs]
    trBatchY = trY[miniBatchIxs]
    r.emb = np.C[trBatchX]
    loss = getLoss2(np, r.emb, trBatchY)
    r.logits =  loss.logits
    r.loss = loss.loss
    return r


def backwardPass(parameters: list[Tensor],
                 loss: Tensor) -> None:
    for p in parameters:
        p.grad = None
    loss.backward()


def backwardPass2(layers: list[Layer],
                  parameters: list[Tensor],
                  loss: Tensor) -> None:
    for l in layers:
        l.out.retain_grad() # for debug only
    for p in parameters:
        p.grad = None
    loss.backward()

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
       p.data += -res.learningRate * p.grad # type: ignore
    return res

class Losses:
    tr: Loss
    val: Loss
    tst: Loss

class Losses2:
    tr: Loss2
    val: Loss2
    tst: Loss2

class Sample:
    values: list[str]
    probs: list[float]
    prob: float


def calcOneProb(probs: list[float]) -> float:
    total = probs[0]
    length = len(probs)
    for p in probs[1:]:
        total *= p ** (1 / length)
    return total


def sampleMany(np: NetParameters,
               cal: CalibrationResult,
               g: torch.Generator,
               contextSize: int,
               itos: dict[int, str],
               countSamples: int,
               maxSampleLength: int) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(countSamples):
        s = sampleOne(np, cal, g, contextSize, itos, maxSampleLength)
        if s == None:
            break
        samples.append(s)
    return samples


def sampleMany2(np: NetParameters2,
               g: torch.Generator,
               contextSize: int,
               itos: dict[int, str],
               countSamples: int,
               maxSampleLength: int) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(countSamples):
        s = sampleOne2(np, g, contextSize, itos, maxSampleLength)
        if s == None:
            break
        samples.append(s)
    return samples


def getProbs(np: NetParameters,
             cal: CalibrationResult,
             context: list[int]) -> Tensor:
    emb = np.C[torch.tensor([context])]
    embCat = emb.view(emb.shape[0], -1)
    hPreActivations = embCat @ np.W1 
    hPreActivations = np.batchNormGain * (hPreActivations - cal.mean) / cal.std + np.batchNormBias
    h = torch.tanh(hPreActivations)
    logits = h @ np.W2 + np.b2
    counts = logits.exp() # counts, equivalent to next character
    probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
    #=probs = F.softmax(logits, dim=1)
    return probs


def getProbs2(np: NetParameters2, context: list[int]) -> Tensor:
    emb = np.C[torch.tensor([context])]
    logits = getLogits2(np, emb)
    probs = F.softmax(logits, dim=1)
    return probs


def sampleOne(np: NetParameters,
           cal: CalibrationResult,
           g: torch.Generator,
           contextSize: int,
           itos: dict[int, str],
           maxLength: int) -> Sample | None:
    s = Sample()
    s.values = []
    s.probs = []
    context = [0] * contextSize
    for i in range(maxLength):
        probs = getProbs(np, cal, context)
        ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
        s.probs.append(probs[0, ix].item())
        context = context[1:] + [ix]
        s.values.append(itos[ix])
        if ix == 0: 
            break
    s.prob = calcOneProb(s.probs)
    return s


def sampleOne2(np: NetParameters2,
           g: torch.Generator,
           contextSize: int,
           itos: dict[int, str],
           maxLength: int) -> Sample | None:
    s = Sample()
    s.values = []
    s.probs = []
    context = [0] * contextSize
    for i in range(maxLength):
        probs = getProbs2(np, context)
        ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
        s.probs.append(probs[0, ix].item())
        context = context[1:] + [ix]
        s.values.append(itos[ix])
        if ix == 0: 
            break
    s.prob = calcOneProb(s.probs)
    return s



def calcProb(np: NetParameters,
             cal: CalibrationResult,
             sample: str,
             contextSize: int,
             stoi: dict[str, int]) -> list[float]:
    ps: list[float] = []
    context = [0] * contextSize
    for i in range(len(sample)):
        probs = getProbs(np, cal, context)
        ix = stoi[sample[i]]
        ps.append(probs[0, ix].item())
        context = context[1:] + [ix]
    return ps
