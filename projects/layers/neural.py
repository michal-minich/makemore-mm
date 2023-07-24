import torch
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
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
    r.logits = getLogits(np.layers, emb)
    r.loss = F.cross_entropy(r.logits, y)#.long())
    return r


def getLogits(layers: list[Layer], emb: Tensor) -> Tensor:
    logits = emb.view(emb.shape[0], -1)
    for l in layers:
        logits = l(logits)
    return logits


class ForwardPassResult(Loss):
    emb: Tensor


def forwardPass(np: NetParameters,
                trX: Tensor,
                trY: Tensor,                
                miniBatchIxs: Tensor) -> ForwardPassResult:
    r = ForwardPassResult()
    batchX = trX[miniBatchIxs]
    batchY = trY[miniBatchIxs]
    r.emb = np.C[batchX]
    loss = getLoss(np, r.emb, batchY)
    r.logits = loss.logits
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
    logits = getLogits(np.layers, emb)
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


def calcProb(np: NetParameters,
             sample: str,
             contextSize: int,
             stoi: dict[str, int]) -> list[float]:
    ps: list[float] = []
    context = [0] * contextSize
    for i in range(len(sample)):
        probs = getProbs(np, context)
        ix = stoi[sample[i]]
        ps.append(probs[0, ix].item())
        context = context[1:] + [ix]
    return ps


def plotActivationsDistribution(np: NetParameters, T: type, useGrad = False):
    title = 'Activation distribution - ' + T.__name__ + " (Grad)" if useGrad else ""
    plt.figure(figsize=(20,4))
    legends = []
    logSimple(title)
    for l in np.layers:
        if isinstance(l, T):
            t : Tensor = l.out.grad if useGrad else l.out # type: ignore
            log("  " + l.name, f"mean: {t.mean():+.5f}, std: {t.std():+.5f}, saturated: {(t.abs() > 0.97).float().mean() * 100:.2f}")
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer ({l.name})')
    plt.legend(legends);
    plt.title(title);
