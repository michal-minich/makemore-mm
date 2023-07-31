from abc import abstractmethod
from torch import Tensor
from mm.common import getSizeString


layerCount = 0


class Layer:
    
    out: Tensor
    name: str

    def __init__(self) -> None:
        global layerCount 
        layerCount += 1
        self.name = type(self).__name__ + " " + str(layerCount)
        
    def longName(self):
        p = self.paramsShapeStr()
        return self.name + ("" if len(p) == 0 else " " + p)
        
    def paramsShapeStr(self):
        return ", ".join(getSizeString(p.shape) for p in self.parameters())

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        pass
    
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass
