import math

import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

def f(x1, x2): # ⽬标函数
    return x1 ** 2 + 2 * x2 ** 2
def f_grad(x1, x2): # ⽬标函数的梯度
    return 2 * x1, 4 * x2
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulatenoisygradient
    g1 += torch.normal(0.0, 1, (1,))
    g2 += torch.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
    return 1

eta = 0.1
lr = constant_lr # Constantlearningrate
# d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))

def exponential_lr():
    # Globalvariablethatisdefinedoutsidethisfunctionandupdatedinside
    global t
    t += 1
    return math.exp(-0.1 * t)
t = 1
lr = exponential_lr
# d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))

def polynomial_lr():
    # Globalvariablethatisdefinedoutsidethisfunctionandupdatedinside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)
t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))




plt.show()
