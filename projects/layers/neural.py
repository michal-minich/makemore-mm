import torch
import torch.nn.functional as F
from torch import Tensor
from mm.printing import *
from mm.common import *
from mm.neural.neural import *
from mm.neural.data import *
from layers import *
from network import *


class Loss:
    logits: Tensor
    loss: Tensor
    

def getLoss(np: NetParameters,
            emb: Tensor,
            y: Tensor) -> Loss:
    r = Loss()
    r.logits = getLogits(np, emb)
    r.loss = F.cross_entropy(r.logits, y)#.long())
    return r


def getLogits(np: NetParameters, emb: Tensor) -> Tensor:
    logits = emb.view(emb.shape[0], -1)
    for l in np.layers:
        logits = l(logits)
    return logits


class ForwardPassResult(Loss):
    emb: Tensor


def forwardPass(np: NetParameters,
                trX: Tensor,
                trY: Tensor,                
                miniBatchIxs: Tensor) -> ForwardPassResult:
    r = ForwardPassResult()
    trBatchX = trX[miniBatchIxs]
    trBatchY = trY[miniBatchIxs]
    r.emb = np.C[trBatchX]
    loss = getLoss(np, r.emb, trBatchY)
    r.logits =  loss.logits
    r.loss = loss.loss
    return r


def backwardPass(layers: list[Layer],
                  parameters: list[Tensor],
                  loss: Tensor) -> None:
    for l in layers:
        l.out.retain_grad() # for debug only
    for p in parameters:
        p.grad = None
    loss.backward()


class Losses:
    tr: Loss
    val: Loss
    tst: Loss
    

def sampleMany(np: NetParameters,
               g: torch.Generator,
               contextSize: int,
               itos: dict[int, str],
               countSamples: int,
               maxSampleLength: int) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(countSamples):
        s = sampleOne(np, g, contextSize, itos, maxSampleLength)
        if s == None:
            break
        samples.append(s)
    return samples


def getProbs(np: NetParameters, context: list[int]) -> Tensor:
    emb = np.C[torch.tensor([context])]
    logits = getLogits(np, emb)
    probs = F.softmax(logits, dim=1)
    return probs


def sampleOne(np: NetParameters,
              g: torch.Generator,
              contextSize: int,
              itos: dict[int, str],
              maxLength: int) -> Sample | None:
    s = Sample()
    s.values = []
    s.probs = []
    context = [0] * contextSize
    for i in range(maxLength):
        probs = getProbs(np, context)
        ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
        s.probs.append(probs[0, ix].item())
        context = context[1:] + [ix]
        s.values.append(itos[ix])
        if ix == 0: 
            break
    s.prob = calcOneProb(s.probs)
    return s
