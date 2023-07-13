from abc import abstractmethod
from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor
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


def buildDataSet(words: list[str], 
                 contextSize: int, 
                 stoi: dict[str, int], 
                 itos: dict[int, str],
                 dtype: torch.dtype,
                 dvc: torch.device) -> tuple[Tensor, Tensor]:
    x: list[list[int]] = []
    y: list[int] = []
    for w in words:
        context = [0] * contextSize
        for ch in w + ".":
            ix = stoi[ch]
            x.append(context)
            y.append(ix)
            #\print("".join(itos[i] for i in context), "--->", itos[ix])
            context = context[1:] + [ix]
    return torch.tensor(x, dtype=dtype, device=dvc), torch.tensor(y, dtype=dtype, device=dvc)

layerCount = 0

class Layer:
    
    out: Tensor
    name: str

    def __init__(self) -> None:
        global layerCount 
        layerCount += 1
        self.name = type(self).__name__ + " " + str(layerCount)

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        pass
    
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass


class Linear(Layer):

    def __init__(self: 'Linear',
                 fanIn: int, 
                 fanOut: int,
                 generator: torch.Generator,
                 dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.weight: Tensor = torch.rand(
            (fanIn, fanOut), generator=generator, dtype=dtype, device=device) / fanIn ** 0.5


    def __call__(self: 'Linear', x: Tensor | float) -> Tensor:
        self.out = x @ self.weight
        return self.out


    def parameters(self: 'Linear') -> list[Tensor]:
        return [self.weight]


class LinearWithBias(Linear):

    def __init__(self: 'LinearWithBias',
                 fanIn: int, 
                 fanOut: int,
                 generator: torch.Generator,
                 dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__(fanIn, fanOut, generator, dtype, device)
        self.bias = torch.zeros(fanOut, dtype=dtype, device=device)


    def __call__(self: 'LinearWithBias', x: Tensor) -> Tensor:
        super().__call__(x)
        self.out += self.bias
        return self.out


    def parameters(self: 'LinearWithBias') -> list[Tensor]:
        return [self.weight, self.bias]


class BatchNorm1d(Layer):

    def __init__(self : 'BatchNorm1d',
                 dim: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 eps=1e-5,
                 momentum=0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.updateRunning = True
        # parameters (trained with back-propagation)
        self.gamma = torch.ones(dim, dtype=dtype, device=device)
        self.beta = torch.zeros(dim, dtype=dtype, device=device)
        # parameters (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim, dtype=dtype, device=device)
        self.running_var = torch.ones(dim, dtype=dtype, device=device)


    def __call__(self: 'BatchNorm1d', x: Tensor) -> Tensor:
        # calculate forward pass
        if self.updateRunning:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xchat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xchat + self.beta
        # update the buffers
        if self.updateRunning:
            with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out


    def parameters(self: 'BatchNorm1d') -> list[Tensor]:
        return [self.gamma, self.beta]


class Tanh(Layer):

    def __call__(self: 'Tanh', x: Tensor) -> Tensor:
        super().__init__()
        self.out = torch.tanh(x)
        return self.out


    def parameters(self):
        return []


class NetParameters:
    C: Tensor
    W1: Tensor
    #b1: Tensor
    W2: Tensor
    b2: Tensor
    batchNormGain: Tensor
    batchNormBias: Tensor
    all: list[Tensor]


class NetParameters2:
    C: Tensor
    layers: list[Layer]
    parameters: list[Tensor]


def makeNetwork(g: torch.Generator, 
                vocabularyLength: int, 
                embeddingDims: int, 
                contextSize: int, 
                hiddenLayerSize: int,
                dvc: torch.device) -> NetParameters:

    fanIn = embeddingDims * contextSize
    W1ratio = (5 / 3) / (fanIn ** 0.5) # 0.2
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



def makeNetwork2(g: torch.Generator, 
                vocabularyLength: int, 
                embeddingDims: int, 
                contextSize: int, 
                hiddenLayerSize: int,
                dtype: torch.dtype,
                dvc: torch.device) -> NetParameters2:
    np = NetParameters2()
    np.C = torch.rand((vocabularyLength, embeddingDims), generator=g)
    firstLayer = LinearWithBias(embeddingDims * contextSize, hiddenLayerSize, g, dtype, dvc)
    np.layers = [
        firstLayer,
        Tanh(),
        LinearWithBias(hiddenLayerSize, hiddenLayerSize, g, dtype, dvc),
        #Tanh(),
        #LinearWithBias(hiddenLayerSize, hiddenLayerSize, g, dtype, dvc),
        #Tanh(),
        #LinearWithBias(hiddenLayerSize, hiddenLayerSize, g, dtype, dvc),
        #Tanh(),
        #LinearWithBias(hiddenLayerSize, hiddenLayerSize, g, dtype, dvc),
        #Tanh(),
        #LinearWithBias(hiddenLayerSize, vocabularyLength, g, dtype, dvc),
    ]
    with torch.no_grad():
        # last layer: make less confident
        firstLayer.weight *= 0.1
        # all other layers: apply gain
        for l in np.layers[:-1]:
            if (isinstance(l, Linear)):
                l.weight *= 5 / 3
    np.parameters = [np.C] + [p for l in np.layers for p in l.parameters()]
    for p in np.parameters:
        p.requires_grad = True
    return np


def printNetwork(np: NetParameters2):
    log('Network Structure')
    for l in np.layers:
        logSimple("Layer " + l.name + ": ", end="")
        for p in l.parameters():
            logSimple(p.shape, end=", ")
        logSimple()
    log('Parameters Count', sum(p.nelement() for p in np.parameters))


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
