import torch
from torch import Tensor
from mm.printing import *
from mm.common import *
from mm.neural.layers import *


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

