import torch
from torch import Tensor


def sToI(chars: list[str]) -> dict[str, int]:
   res =  {s:i+1 for i,s in enumerate(chars)}
   res["."] = 0
   return res;


def sToI2(chars: list[str]) -> dict[str, int]:
   stoi = { "." : 0 }
   for i, ch in enumerate(chars):
       stoi[ch] = i + 1
   return stoi


def iToS(stoi: dict[str, int]) -> dict[int, str]:
    return {i:s for s,i in stoi.items()}


def buildDataSet(words: list[str], 
                 contextSize: int, 
                 stoi: dict[str, int], 
                 itos: dict[int, str],
                 dtype: torch.dtype,
                 dvc: torch.device) -> tuple[Tensor, Tensor]:
    x: list[list[int]] = []
    y: list[int] = []
    for w in words:
        context = [0] * contextSize
        for ch in w + ".":
            ix = stoi[ch]
            x.append(context)
            y.append(ix)
            #\print("".join(itos[i] for i in context), "--->", itos[ix])
            context = context[1:] + [ix]
    return torch.tensor(x, dtype=dtype, device=dvc), torch.tensor(y, dtype=dtype, device=dvc)


