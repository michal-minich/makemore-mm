import torch
from torch import Tensor
from mm.printing import *


class NetParameters:
    C: Tensor
    W1: Tensor
    b1: Tensor
    W2: Tensor
    b2: Tensor
    all: list[Tensor]


def makeNetwork(g: torch.Generator, 
                vocabularyLength: int, 
                embeddingDims: int, 
                contextSize: int, 
                hiddenLayerSize: int,
                dvc: torch.device) -> NetParameters:

    fanIn = embeddingDims * contextSize
    W1ratio = 0.20
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
    np.b1 = b1ratio * torch.randn(hiddenLayerSize, generator = g, device=dvc) 
    np.W2 = W2ratio * torch.randn((hiddenLayerSize, vocabularyLength), generator = g, device=dvc)
    np.b2 = b2ratio * torch.randn(vocabularyLength, generator = g, device=dvc) 
    np.all = [np.C, np.W1, np.b1, np.W2, np.b2]
    for p in np.all:
        p.requires_grad = True
        
    return np
