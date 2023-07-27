from typing import Any
from datetime import datetime
import os
import torch
from matplotlib import pyplot as plt
from mm.common import *
from mm.layers import Layer

logsPath = "logs/"
logFilePath = "logs/log.txt"
plotCounter = 1


def log(
    label: Any | None = "",
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n"
) -> None:
    lbl = "" if label == None else f"{(label + ':'):<24}"
    logSimple(lbl, *values, sep=sep, end=end)


def logSimple(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n"
) -> None:
    vs = [getSizeString(v) if isinstance(v, torch.Size) else v for v in values]
    print(*vs, sep=sep, end=end)
    with open(logFilePath, "a") as f:
        print(*vs, sep=sep, end=end, file=f)


def logSection(title: str) -> None:
    log(title, "-------------------------- " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def initLogging(title: str) -> None:
    now = datetime.now()
    month = now.strftime("%m")
    day = now.strftime("%d")
    time = datetime.now().strftime("%H_%M_%S")
    global logsPath, logFilePath
    logsPath = "./logs/" + month + "/" + day + "/" + time + "/"
    logFilePath = logsPath + "log.txt"
    if not os.path.exists(logsPath):
        os.makedirs(logsPath)
    logSection(title)


def getSizeString(size: torch.Size) -> str: 
    return "[" + ", ".join(str(dim) for dim in size) + "]"


def savePlot() -> None:
    global logsPath, plotCounter
    time = datetime.now().strftime("%H_%M_%S")
    t = plt.gca().get_title().replace("/", "-")
    filename = str(plotCounter).zfill(2) + " " + t + " (" + time + ").png"
    plotCounter += 1
    log("Plot", filename)
    plt.savefig(logsPath + filename)


def plotActivationsDistribution(T: type, layers: list[Layer], useGrad = False):
    title = "Activations distribution - " + T.__name__ + (" (Grad)" if useGrad else "")
    plt.figure(figsize=(15, 7))
    plt.title(title)
    legends = []
    logSimple(title)
    for l in layers:
        if isinstance(l, T):
            t: torch.Tensor = not_null(l.out.grad) if useGrad else l.out
            log("  " + l.name, f"mean: {t.mean():+.5f}, std: {t.std():+.5f}", end="")
            if useGrad:
                logSimple("")
            else:
                logSimple(f", saturated: {(t.abs() > 0.97).float().mean() * 100:.2f}%")
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(l.name)
    plt.legend(legends)
    savePlot()


def plotGradWeightsDistribution(T: type, C: torch.Tensor, layers: list[Layer]):
    title = "Gradients weights distribution"
    plt.figure(figsize=(15, 7))
    legends: list[str] = []
    logSimple(title)
    logSimple(f"  C")
    gradWeightForParam(C, "C", legends)
    for l in layers:
        if isinstance(l, T):
            logSimple(f"  " + l.name, end=("" if len(l.parameters()) == 1 else "\n"))
            for p in l.parameters():
                gradWeightForParam(p, l.name, legends)
    plt.legend(legends)
    plt.title(title)
    savePlot()


def gradWeightForParam(p: torch.Tensor, layerName: str, legends: list[str]):
    g = not_null(p.grad)
    log(f"    Weight", f"{getSizeString(p.shape):>10}, mean: {g.mean():+.5f}, std: {g.std():e}", end="")
    logSimple(f", data ratio: {g.std() / p.std():e}")
    hy, hx = torch.histogram(g, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{layerName} {tuple(p.shape)}')


def plotGradientUpdateRatio(ud: list[float], parameters: list[torch.Tensor], names: list[str]) -> None:
    title = "Gradient update / Data ratio"
    plt.figure(figsize=(15, 7))
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2:
            plt.plot([ud[j][i] for j in range(len(ud))])
            legends.append(names[i])
    plt.plot([0, len(ud)], [-3, -3], "k") # these ratios should be ~1e-3, indicate on plot
    plt.legend(legends);
    plt.title(title);
    savePlot();


def plotLearningRate(lrAtIx: list[float]) -> None:
    log("Max learning rate", max(lrAtIx))
    log("Min learning rate", min(lrAtIx))
    plt.figure()
    plt.plot(range(len(lrAtIx)), lrAtIx, "black")
    plt.title('Learning rate progress');
    plt.ylim(min(lrAtIx), max(lrAtIx))
    plt.grid(True)
    savePlot()
    

def plotXY(x: list[float], y: list[float], title: str) -> None:
    fig, ax = plt.subplots()
    fig.set_facecolor("#777777")
    ax.set_facecolor("#222222")
    ax.plot(x, y)
    plt.title(title);
    savePlot()
    

def plotEmb(C: torch.Tensor, itos: dict[int, str], dim: int) -> None:
    fig = plt.figure(figsize=(10, 10))
    plt.title(f"Embedding at [{dim}, {dim+1}]");
    fig.set_facecolor("#777777")
    sc = plt.scatter(C[:, dim].data, C[:,dim + 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, dim].item(), C[i, dim + 1].item(), itos[i], ha="center", va="center", color="white")
    plt.grid()
    savePlot()
