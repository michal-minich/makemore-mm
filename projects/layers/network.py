import torch
from torch import Tensor
from mm.printing import *
from mm.common import *
from layers import *


class NetParameters:
    C: Tensor
    layers: list[Layer]
    parameters: list[Tensor]


def makeNetwork(g: torch.Generator, 
                vocabularyLength: int, 
                embeddingDims: int, 
                contextSize: int, 
                hiddenLayerSize: int,
                dtype: torch.dtype,
                dvc: torch.device) -> NetParameters:
    np = NetParameters()
    np.C = torch.rand((vocabularyLength, embeddingDims), generator=g)
    firstLayer = LinearWithBias(embeddingDims * contextSize, hiddenLayerSize, g, dtype, dvc)

    np.layers = [
        firstLayer,
        #BatchNorm1d(hiddenLayerSize, dtype, dvc),
        Tanh(),
        LinearWithBias(hiddenLayerSize, hiddenLayerSize, g, dtype, dvc),
        #BatchNorm1d(hiddenLayerSize, dtype, dvc),
        Tanh(),
        LinearWithBias(hiddenLayerSize, hiddenLayerSize, g, dtype, dvc),
        #BatchNorm1d(hiddenLayerSize, dtype, dvc),
        Tanh(),
        LinearWithBias(hiddenLayerSize, hiddenLayerSize, g, dtype, dvc),
        #BatchNorm1d(hiddenLayerSize, dtype, dvc),
        Tanh(),
        LinearWithBias(hiddenLayerSize, hiddenLayerSize, g, dtype, dvc),
        #BatchNorm1d(hiddenLayerSize, dtype, dvc),
        Tanh(),
        #BatchNorm1d(vocabularyLength, dtype, dvc),
    ]    
    
    lastLayer = LinearWithBias(hiddenLayerSize, vocabularyLength, g, dtype, dvc)

    np.layers.append(lastLayer)


    with torch.no_grad():
        # last layer: make less confident
        lastLayer.weight *= 0.1
        # all other layers: apply gain
        for l in np.layers[:-1]:
            if (isinstance(l, Linear) or isinstance(l, LinearWithBias)):
                l.weight *= 5 / 3
    np.parameters = [np.C] + [p for l in np.layers for p in l.parameters()]
    for p in np.parameters:
        p.requires_grad = True
    return np


def printNetworkInfo(np: NetParameters):
    log('Network Structure')
    for l in np.layers:
        logSimple("Layer " + l.name + ": ", end="")
        for p in l.parameters():
            logSimple(p.shape, end="; ")
        logSimple()
    log('Parameters Count', sum(p.nelement() for p in np.parameters))

