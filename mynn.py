import torch
import torch.nn.functional as F
from torch import Tensor
from datetime import datetime
import os


logFilePath = "log.txt"


def log(
    label: str | None = "",
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n"
) -> None:
    lbl = "" if label == None else f"{(label + ':'):<20}"
    logSimple(lbl, *values, sep=sep, end=end)


def logSimple(
    label: str | None = "",
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


def findLowestIndex(arr: list) -> int:
    ix = 0
    for i in range(1, len(arr)):
        if arr[i] < arr[ix]:
            ix = i
    return ix

def readFileSplitByLine(name: str) -> list[str]:
    words = open(name, "r", encoding="utf-8").read().splitlines()
    return words


def sToI(chars: list[str]) -> dict[str, int]:
   res =  {s:i+1 for i,s in enumerate(chars)}
   res["."] = 0
   return res;

def sToI2(chars: list[str]) -> dict[str, int]:
   stoi = { "." : 0 }
   for i, ch in enumerate(chars):
       stoi[ch] = i + 1
   return stoi


def iToS(stoi: dict[str, int]) -> dict[int, str]:
    return {i:s for s,i in stoi.items()}

device = torch.device("cuda")
def buildDataSet(words: list[str], 
                 contextSize: int, 
                 stoi: dict[str, int], 
                 itos: dict[int, str],
                 dvc: torch.device) -> tuple[Tensor, Tensor]:
    X: list[list[int]] = []
    Y: list[int] = []
    for w in words:
        context = [0] * contextSize
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            #\print("".join(itos[i] for i in context), "--->", itos[ix])
            context = context[1:] + [ix]
    return torch.tensor(X, device=dvc), torch.tensor(Y, device=dvc)


class NetParameters:
    C: Tensor
    W1: Tensor
    #b1: Tensor
    W2: Tensor
    b2: Tensor
    batchNormGain: Tensor
    batchNormBias: Tensor
    all: list[Tensor]


def makeNetwork(g: torch.Generator, 
                vocabularyLength: int, 
                embeddingDims: int, 
                contextSize: int, 
                hiddenLayerSize: int,
                dvc: torch.device) -> NetParameters:

    fanIn = embeddingDims * contextSize
    W1ratio = 0.2 # (5 / 3) / (fanIn ** 0.5)
    log('W1ratio', W1ratio)
    b1ratio = 0.01
    log('b1ratio', b1ratio)
    W2ratio = 0.1
    log('W2ratio', W2ratio)
    b2ratio = 0
    log('b2ratio', b2ratio)

    np = NetParameters()
    np.C =  torch.randn((vocabularyLength, embeddingDims), generator = g, device=dvc)
    np.W1 = W1ratio * torch.randn((fanIn, hiddenLayerSize), generator = g, device=dvc) 
    #np.b1 = b1ratio * torch.randn(hiddenLayerSize, generator = g, device=dvc) 
    np.W2 = W2ratio * torch.randn((hiddenLayerSize, vocabularyLength), generator = g, device=dvc)
    np.b2 = b2ratio * torch.randn(vocabularyLength, generator = g, device=dvc) 
    np.batchNormGain = torch.ones((1, hiddenLayerSize))
    np.batchNormBias = torch.zeros((1, hiddenLayerSize))
    np.all = [np.C, np.W1,# np.b1, 
               np.W2, np.b2, np.batchNormGain, np.batchNormBias]
    for p in np.all:
        p.requires_grad = True
        
    return np


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


class ForwardPassResult(Loss):
    emb: Tensor


def forwardPass(np: NetParameters,
                cal: CalibrationResult,
                trX: Tensor,
                trY: Tensor,                
                miniBatchIxs: Tensor) -> ForwardPassResult:
    r = ForwardPassResult()
    r.emb = np.C[trX[miniBatchIxs]]
    loss = getLoss(np, cal, r.emb, trY[miniBatchIxs])
    r.hPreActivations = loss.hPreActivations
    r.h = loss.h
    r.logits = loss.logits
    r.loss = loss.loss
    return r


def backwardPass(parameters: list[Tensor],
                 loss: Tensor) -> None:
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

class Sample:
    values: list[str]
    probs: list[float]
    prob: float

def sample(np: NetParameters,
           cal: CalibrationResult,
           g: torch.Generator,
           contextSize: int,
           itos: dict[int, str],
           countSamples: int) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(countSamples):
        values: list[int] = []
        s = Sample()
        samples.append(s)
        s.values = []
        s.probs = []
        context = [0] * contextSize
        while True:
            emb = np.C[torch.tensor([context])] 
            hPreActivations = emb.view(emb.shape[0], -1) @ np.W1 
            hPreActivations = np.batchNormGain * (hPreActivations - cal.mean) / cal.std + np.batchNormBias
            h = torch.tanh(hPreActivations)
            logits = h @ np.W2 + np.b2
            counts = logits.exp() # counts, equivalent to next character
            probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
            #=probs = F.softmax(logits, dim=1)
            ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
            s.probs.append(probs[0, ix].item() / (1/27) )
            context = context[1:] + [ix]
            values.append(ix)
            s.values.append(itos[ix])
            if ix == 0: 
                break
        s.prob = calcOneProb(s.probs)
    return samples


def sample2(np: NetParameters,
            cal: CalibrationResult,
            g: torch.Generator,
            contextSize: int,
            itos: dict[int, str],
            countSamples: int) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(countSamples):
        values: list[int] = []
        s = Sample()
        samples.append(s)
        s.values = []
        s.probs = []
        probs2: list[float] = []
        context = [0] * contextSize
        while True:
            emb = np.C[torch.tensor([context])]
            hPreActivations = emb.view(emb.shape[0], -1) @ np.W1
            hPreActivations = np.batchNormGain * (hPreActivations - cal.mean) / cal.std + np.batchNormBias
            h = torch.tanh(hPreActivations)
            logits = h @ np.W2 + np.b2
            counts = logits.exp() # counts, equivalent to next character
            probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
            #probs = F.softmax(logits, dim=1)
            ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
            probs2.append(probs[0, ix].item())
            s.probs.append(probs[0, ix].item() / (1/27) )
            context = context[1:] + [ix]
            values.append(ix)
            s.values.append(itos[ix])
            if ix == 0: 
                break
        s.prob = calcOneProb(probs2)
    return samples


def calcOneProb(probs: list[float]) -> float:
    total = probs[0]
    for p in probs[1:]:
        total *= p
    return total / len(probs)


def calcProb(np: NetParameters,
             cal: CalibrationResult,
             sample: str,
             contextSize: int,
             stoi: dict[str, int]) -> list[float]:
    values: list[int] = []
    ps: list[float] = []
    probs2: list[float] = []
    context = [0] * contextSize
    for i in range(len(sample)):
        emb = np.C[torch.tensor([context])]    
        hPreActivations = emb.view(emb.shape[0], -1) @ np.W1
        hPreActivations = np.batchNormGain * (hPreActivations - cal.mean) / cal.std + np.batchNormBias
        h = torch.tanh(hPreActivations)
        logits = h @ np.W2 + np.b2
        counts = logits.exp() 
        probs = counts / counts.sum(1, keepdim=True)
        ix = stoi[sample[i]]
        ps.append(probs[0, ix].item() / (1/27) )
        context = context[1:] + [ix]
    return ps
