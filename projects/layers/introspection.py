from mm.printing import *
from mm.common import *
from layers import *
from network import *


def printNetwork(np: NetParameters2):
    log('Network Structure')
    for l in np.layers:
        logSimple("Layer " + l.name + ": ", end="")
        for p in l.parameters():
            logSimple(p.shape, end=", ")
        logSimple()
    log('Parameters Count', sum(p.nelement() for p in np.parameters))
