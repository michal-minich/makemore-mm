import torch
import torch.nn.functional as F
from torch import Tensor
from mm.printing import *
from mm.common import *
from mm.neural.neural import *
from mm.neural.data_set import *
from layers import *
from network import *


class Loss2:
    logits: Tensor
    loss: Tensor
    

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


class ForwardPassResult2(Loss2):
    emb: Tensor


def forwardPass2(np: NetParameters2,
                 trX: Tensor,
                 trY: Tensor,                
                 miniBatchIxs: Tensor) -> ForwardPassResult2:
    r = ForwardPassResult2()
    trBatchX = trX[miniBatchIxs]
    trBatchY = trY[miniBatchIxs]
    r.emb = np.C[trBatchX]
    loss = getLoss2(np, r.emb, trBatchY)
    r.logits =  loss.logits
    r.loss = loss.loss
    return r


def backwardPass2(layers: list[Layer],
                  parameters: list[Tensor],
                  loss: Tensor) -> None:
    for l in layers:
        l.out.retain_grad() # for debug only
    for p in parameters:
        p.grad = None
    loss.backward()


class Losses2:
    tr: Loss2
    val: Loss2
    tst: Loss2
    

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


def getProbs2(np: NetParameters2, context: list[int]) -> Tensor:
    emb = np.C[torch.tensor([context])]
    logits = getLogits2(np, emb)
    probs = F.softmax(logits, dim=1)
    return probs


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
