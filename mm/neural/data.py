import torch
from torch import Tensor
from mm.common import *
from mm.printing import *


class LoadDataResult:
    words: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]


def loadData(filePath: str) -> LoadDataResult:
    ldr = LoadDataResult()
    ldr.words = readFileSplitByLine(filePath)
    originalVocabulary = sorted(list(set("".join(ldr.words))))
    ldr.stoi = sToI(originalVocabulary)
    ldr.itos = iToS(ldr.stoi)
    return ldr


def printDataInfo(ldr: LoadDataResult) -> None:
    log("First few words", ldr.words[:5])
    log("Words counts", len(ldr.words))
    log("Vocabulary", list(ldr.stoi.keys()))
    log("stoi", ldr.stoi)
    log("itos", ldr.itos)
    vocabularyLength = len(ldr.stoi)
    log("Vocabulary + end length", vocabularyLength)
    log("random probability", f"{-torch.tensor(1 / vocabularyLength).log().item():.4f}")


def sToI2(chars: list[str]) -> dict[str, int]:
   res =  { s : i + 1 for i, s in enumerate(chars) }
   res["."] = 0
   return res;


def sToI(chars: list[str]) -> dict[str, int]:
   stoi = { "." : 0 }
   for i, ch in enumerate(chars):
       stoi[ch] = i + 1
   return stoi


def iToS(stoi: dict[str, int]) ->  dict[int, str]:
    return { i: s for s, i in stoi.items() }


class DataSetSection:
    x: Tensor
    y: Tensor


class DataSet:
    tr: DataSetSection
    val: DataSetSection
    tst: DataSetSection


def buildDataSet(ldr: LoadDataResult,
                 contextSize: int,
                 trRatio: float,
                 valRatio: float,
                 dtype: torch.dtype,
                 dvc: torch.device) -> DataSet:
    
    log("Data dtype", dtype)
    log("Training ratio", trRatio)
    log("Validation ratio", valRatio)

    ds = DataSet()

    lenWords = len(ldr.words);

    lenTrain = int(trRatio * lenWords)
    trWords = ldr.words[:lenTrain]
    ds.tr = buildDataSetSection(trWords, contextSize, ldr.stoi, ldr.itos, dtype, dvc)
    log("Training", "length", lenTrain, "shape", ds.tr.x.shape, trWords[:3])

    endVal = int(valRatio * lenWords)
    valWords = ldr.words[lenTrain:endVal];
    ds.val = buildDataSetSection(valWords, contextSize, ldr.stoi, ldr.itos, dtype, dvc)
    log("Validation", "length", endVal - lenTrain, "shape",  ds.val.x.shape, valWords[:3])

    lenTest = lenWords - endVal
    tstWords = ldr.words[endVal:]
    ds.tst = buildDataSetSection(tstWords, contextSize, ldr.stoi, ldr.itos, dtype, dvc)
    log("Test", "length", lenTest, "shape", ds.tst.x.shape,  tstWords[:3])
    return ds


def buildDataSetSection(words: list[str], 
                        contextSize: int, 
                        stoi: dict[str, int], 
                        itos: dict[int, str],
                        dtype: torch.dtype,
                        dvc: torch.device) -> DataSetSection:
    x: list[list[int]] = []
    y: list[int] = []
    dss = DataSetSection()
    for w in words:
        context = [0] * contextSize
        for ch in w + ".":
            ix = stoi[ch]
            x.append(context)
            y.append(ix)
            #\print("".join(itos[i] for i in context), "--->", itos[ix])
            context = context[1:] + [ix]
    dss.x = torch.tensor(x,device=dvc)
    dss.y = torch.tensor(y,  device=dvc)
    return dss
