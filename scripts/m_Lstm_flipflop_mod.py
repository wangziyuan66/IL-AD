from taiyaki.activation import swish
from taiyaki.layers import (
    Convolution, Lstm, Reverse, Serial, GlobalNormFlipFlop)
import torch.nn as nn

def network(insize=1, size=256, winlen=19, stride=5, alphabet_info=None):
    nbase = 4 if alphabet_info is None else alphabet_info.ncan_base
    winlen2 = 5
    Linear = nn.Linear(size,20)
    Activation = nn.ReLU()
    return Serial([
        Convolution(insize, 4, winlen2, stride=1, fun=swish),
        Convolution(4, 16, winlen2, stride=1, fun=swish),
        Convolution(16, size, winlen, stride=stride, fun=swish),
        Reverse(Lstm(size, size)),
        Lstm(size, size),
        Reverse(Lstm(size, size)),
        Lstm(size, size),
        Reverse(Lstm(size, size)),
        nn.Sequential(Linear,Activation)
    ])
