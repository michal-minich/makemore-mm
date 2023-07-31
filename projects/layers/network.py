import torch
from torch import Tensor
from mm.layers import *
from mm.printing import *
from mm.common import *
from layers import *


class NetParameters:
    C: Tensor
    layers: list[Layer]
    parameters: list[Tensor]
    paramNames: list[str]


def makeNetwork(g: torch.Generator, 
                vocabularyLength: int, 
                embeddingDims: int, 
                contextSize: int, 
                hiddenLayerSize: int,
                dtype: torch.dtype,
                dvc: torch.device) -> NetParameters:
    np = NetParameters()
    np.C = torch.rand((vocabularyLength, embeddingDims), generator=g)
    h = hiddenLayerSize

    np.layers = [
        Linear(embeddingDims * contextSize, h, g, dtype, dvc),
        BatchNorm1d(h, dtype, dvc),
        Tanh(),

        Linear(h, h, g, dtype, dvc),
        BatchNorm1d(h, dtype, dvc),
        Tanh(),

        Linear(h, h, g, dtype, dvc),
        BatchNorm1d(h, dtype, dvc),
        Tanh(),
        
        Linear(h, h, g, dtype, dvc),
        BatchNorm1d(h, dtype, dvc),
        Tanh(),

        Linear(h, h, g, dtype, dvc),
        BatchNorm1d(h, dtype, dvc),
        Tanh(),

        Linear(h, vocabularyLength, g, dtype, dvc)
    ]    
    
    lastLayer = BatchNorm1d(vocabularyLength, dtype, dvc)

    np.layers.append(lastLayer)


    with torch.no_grad():
        # last layer: make less confident
        lastLayer.gamma *= 0.1
    np.parameters = [np.C] + [p for l in np.layers for p in l.parameters()]
    np.paramNames = ["C"] + [l.name for l in np.layers for p in l.parameters()]
    for p in np.parameters:
        p.requires_grad = True
    return np


def makeNetwork4(g: torch.Generator, 
                vocabularyLength: int, 
                embeddingDims: int, 
                contextSize: int, 
                hiddenLayerSize: int,
                dtype: torch.dtype,
                dvc: torch.device) -> NetParameters:
    np = NetParameters()
    np.C = torch.rand((vocabularyLength, embeddingDims), generator=g)
    h = hiddenLayerSize

    np.layers = [
        Linear(embeddingDims * contextSize, h, g, dtype, dvc),
        BatchNorm1d(h, dtype, dvc),
        Tanh(),

        Linear(h, vocabularyLength, g, dtype, dvc)
    ]    
    
    lastLayer = BatchNorm1d(vocabularyLength, dtype, dvc)

    np.layers.append(lastLayer)


    with torch.no_grad():
        # last layer: make less confident
        lastLayer.gamma *= 0.1
    np.parameters = [np.C] + [p for l in np.layers for p in l.parameters()]
    np.paramNames = ["C"] + [l.name for l in np.layers for p in l.parameters()]
    for p in np.parameters:
        p.requires_grad = True
    return np


def makeNetwork2(g: torch.Generator, 
                vocabularyLength: int, 
                embeddingDims: int, 
                contextSize: int, 
                hiddenLayerSize: int,
                dtype: torch.dtype,
                dvc: torch.device) -> NetParameters:
    np = NetParameters()
    np.C = torch.rand((vocabularyLength, embeddingDims), generator=g)
    h = hiddenLayerSize

    np.layers = [
        LinearWithBias(embeddingDims * contextSize, h, g, dtype, dvc),
        Tanh(),

        LinearWithBias(h, h, g, dtype, dvc),
        Tanh(),

        LinearWithBias(h, h, g, dtype, dvc),
        Tanh(),
        
        LinearWithBias(h, h, g, dtype, dvc),
        Tanh(),

        LinearWithBias(h, h, g, dtype, dvc),
        Tanh(),

    ]    
    
    lastLayer = LinearWithBias(h, vocabularyLength, g, dtype, dvc)

    np.layers.append(lastLayer)


    with torch.no_grad():
        # last layer: make less confident
        lastLayer.weight *= 0.1
        # all other layers: apply gain
        for l in np.layers[:-1]:
            if (isinstance(l, Linear)):
                l.weight *= 5 / 3
    np.parameters = [np.C] + [p for l in np.layers for p in l.parameters()]
    np.paramNames = ["C"] + [l.name for l in np.layers for p in l.parameters()]
    for p in np.parameters:
        p.requires_grad = True
    return np


def makeNetwork3(g: torch.Generator, 
                vocabularyLength: int, 
                embeddingDims: int, 
                contextSize: int, 
                hiddenLayerSize: int,
                dtype: torch.dtype,
                dvc: torch.device) -> NetParameters:
    np = NetParameters()
    np.C = torch.rand((vocabularyLength, embeddingDims), generator=g)
    h = hiddenLayerSize

    np.layers = [
        LinearWithBias(embeddingDims * contextSize, h, g, dtype, dvc),
        LinearWithBias(h, h, g, dtype, dvc),
        LinearWithBias(h, h, g, dtype, dvc),        
        LinearWithBias(h, h, g, dtype, dvc),
        LinearWithBias(h, h, g, dtype, dvc),
    ]    
    
    lastLayer = LinearWithBias(h, vocabularyLength, g, dtype, dvc)

    np.layers.append(lastLayer)

    with torch.no_grad():
        # last layer: make less confident
        lastLayer.weight *= 0.1
    np.parameters = [np.C] + [p for l in np.layers for p in l.parameters()]
    np.paramNames = ["C"] + [l.name for l in np.layers for p in l.parameters()]
    for p in np.parameters:
        p.requires_grad = True
    return np


def printNetworkInfo(np: NetParameters):
    log('Network Layers Structure')
    for l in np.layers:
        log("  " + l.name, l.paramsShapeStr())
    log('Parameters Count', sum(p.nelement() for p in np.parameters))

