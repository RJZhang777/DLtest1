import torch
from torch import nn
from d2l import torch as d2l

# ⽣产⽹络的⼯⼚模式
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2))
    return net
x = torch.randn(size=(1, 512))
net = get_net()
net = torch.jit.script(net)

#@save
class Benchmark:
    """⽤于测量运⾏时间"""
    def __init__(self, description='Done'):
        self.description = description
    def __enter__(self):
        self.timer = d2l.Timer()
        return self
    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

# net = get_net()
# with Benchmark('⽆torchscript'):
#     for i in range(1000): net(x)
net = torch.jit.script(net)
with Benchmark('有torchscript'):
    for i in range(1000): net(x)

net.save('my_mlp')
