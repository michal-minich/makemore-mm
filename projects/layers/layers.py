from abc import abstractmethod
import torch
from torch import Tensor
from mm.printing import *


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
        self.weight: Tensor = torch.randn(
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
        self.training = True
        # parameters (trained with back-propagation)
        self.gamma = torch.ones(dim, dtype=dtype, device=device)
        self.beta = torch.zeros(dim, dtype=dtype, device=device)
        # parameters (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim, dtype=dtype, device=device)
        self.running_var = torch.ones(dim, dtype=dtype, device=device)


    def __call__(self: 'BatchNorm1d', x: Tensor) -> Tensor:
        # calculate forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xchat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xchat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out


    def parameters(self: 'BatchNorm1d') -> list[Tensor]:
        return [self.gamma, self.beta]


class Tanh(Layer):

    def __init__(self : 'Tanh') -> None:
        super().__init__()


    def __call__(self: 'Tanh', x: Tensor) -> Tensor:
        self.out = torch.tanh(x)
        return self.out


    def parameters(self):
        return []
