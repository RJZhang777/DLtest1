import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l

def f(x):
    return x * torch.cos(np.pi * x)
def g(x):
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)

def annotate(text, xy, xytext): #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,arrowprops=dict(arrowstyle='->'))
# x = torch.arange(0.5, 1.5, 0.01)
# d2l.set_figsize((4.5, 2.5))
# d2l.plot(x, [f(x), g(x)], 'x', 'risk')
# annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
# annotate('min of risk', (1.1, -1.05), (0.95, -0.5))

# x = torch.arange(-1.0, 2.0, 0.01)
# d2l.plot(x, [f(x), ], 'x', 'f(x)')
# annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
# annotate('global minimum', (1.1, -0.95), (0.6, 0.8))

# x = torch.arange(-2.0, 2.0, 0.01)
# d2l.plot(x, [x**3], 'x', 'f(x)')
# annotate('saddle point', (0, -0.2), (-0.52, -5.0))

# x, y = torch.meshgrid(torch.linspace(-1.0, 1.0, 101), torch.linspace(-1.0, 1.0, 101))
# z = x**2 - y**2
# ax = d2l.plt.figure().add_subplot(111, projection='3d')
# ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
# ax.plot([0], [0], [0], 'rx')
# ticks = [-1, 0, 1]
# d2l.plt.xticks(ticks)
# d2l.plt.yticks(ticks)
# ax.set_zticks(ticks)
# d2l.plt.xlabel('x')
# d2l.plt.ylabel('y');

# x = torch.arange(-2.0, 5.0, 0.01)
# d2l.plot(x, [torch.tanh(x)], 'x', 'f(x)')
# annotate('vanishing gradient', (4, 1), (2, 0.0))

f = lambda x: 0.5 * x**2 # 凸函数
g = lambda x: torch.cos(np.pi * x) # ⾮凸函数
h = lambda x: torch.exp(0.5 * x) # 凸函数
x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
# d2l.use_svg_display()
# _, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
# for ax, func in zip(axes, [f, g, h]):
#     d2l.plot([x, segment], [func(x), func(segment)], axes=ax)

f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')

plt.show()
