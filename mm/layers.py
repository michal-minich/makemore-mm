from abc import abstractmethod
from torch import Tensor


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
