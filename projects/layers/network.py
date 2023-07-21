import torch
from torch import Tensor
from mm.printing import *
from mm.common import *
from layers import *


class NetParameters2:
    C: Tensor
    layers: list[Layer]
    parameters: list[Tensor]


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

