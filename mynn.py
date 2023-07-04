import torch
import torch.nn.functional as F
from torch import Tensor

def findLowestIndex(arr: list) -> int:
    ix = 0
    for i in range(1, len(arr)):
        if arr[i] < arr[ix]:
            ix = i
    return ix

def readFileSplitByLine(name: str) -> list[str]:
    words = open(name, 'r', encoding='utf-8').read().splitlines()
    return words


def sToI(chars: list[str]) -> dict[str, int]:
   stoi = { '.' : 0 }
   for i, ch in enumerate(chars):
       stoi[ch] = i + 1
   return stoi


def iToS(stoi: dict[str, int]) -> dict[int, str]:
    return {i:s for s,i in stoi.items()}


def buildDataSet(words: list[str], 
                 contextSize: int, 
                 stoi: dict[str, int], 
                 itos: dict[int, str]) -> tuple[Tensor, Tensor]:
    X: list[list[int]] = []
    Y: list[int] = []
    for w in words:
        context = [0] * contextSize
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            #\print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


class NetParameters:
    C: Tensor
    W1: Tensor
    b1: Tensor
    W2: Tensor
    b2: Tensor
    all: list[Tensor]


def makeNetwork(g: torch.Generator, 
                vocabularySize: int, 
                embeddingSize: int, 
                contextSize: int, 
                hiddenLayerSize: int) -> NetParameters:
    np = NetParameters()
    np.C = torch.randn((vocabularySize, embeddingSize), generator = g)
    np.W1 = torch.randn((embeddingSize * contextSize, hiddenLayerSize), generator = g)
    np.b1 = torch.randn(hiddenLayerSize, generator = g)
    np.W2 = torch.randn((hiddenLayerSize, vocabularySize), generator = g)
    np.b2 = torch.randn(vocabularySize, generator = g) 
    np.all = [np.C, np.W1, np.b1, np.W2, np.b2]
    for p in np.all:
        p.requires_grad = True
    return np


class Loss:
    h: Tensor
    logits: Tensor
    loss: Tensor


class ForwardPassResult(Loss):
    emb: Tensor
    h: Tensor
    logits: Tensor
    loss: Tensor


def forwardPass(np: NetParameters,
                trX: Tensor,
                trY: Tensor,                
                miniBatchIxs: Tensor) -> ForwardPassResult:
    r = ForwardPassResult()
    r.emb = np.C[trX[miniBatchIxs]]
    loss = getLoss(np, r.emb, trY[miniBatchIxs])
    r.h = loss.h
    r.logits = loss.logits  
    r.loss = loss.loss
    return r


#@torch.no_grad()
def getLoss(np: NetParameters,
            emb: Tensor,
            y: Tensor) -> Loss:
    r = Loss()
    embView = emb.view(emb.shape[0], -1)
    r.h = torch.tanh(embView @ np.W1 + np.b1)
    r.logits = r.h @ np.W2 + np.b2
    r.loss = F.cross_entropy(r.logits, y)
    return r


def backwardPass(parameters: list[Tensor],
                 loss: Tensor) -> None:
  for p in parameters:
    p.grad = None
  loss.backward()


def updateNet(parameters: list[Tensor],
              iteration: int,
              learningRate: float):
    #learningRate = lrs[iteration]
    learningRate = 0.1 if iteration < 20_000 else 0.01
    for p in parameters:
       p.data += -learningRate * p.grad # type: ignore


def sample(np: NetParameters,
           g: torch.Generator,
           contextSize: int,
           itos: dict[int, str],
           countSamples: int):
    for _ in range(countSamples):
        out = []
        context = [0] * contextSize
        while True:
            emb = np.C[torch.tensor([context])] # (1,block_size,d)
            h = torch.tanh(emb.view(1, -1) @ np.W1 + np.b1)
            logits = h @ np.W2 + np.b2
            counts = logits.exp() # counts, equivalent to next character
            probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
            #probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0: 
                break
        print(''.join(itos[i] for i in out))
