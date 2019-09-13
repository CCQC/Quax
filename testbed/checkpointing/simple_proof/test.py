import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
#from my_checkpoint import MyCheckpointFunction, mycheckpoint
from clean_checkpoint import MyCheckpointFunction, mycheckpoint

class Test1(nn.Module):
    def __init__(self):
        super(Test1, self).__init__()

    def bottleneck(self, inp):
        out = torch.sum(torch.sqrt(inp))
        return out

    def bottleneck2(self, inp):
        out = inp**2 + inp**3 - 2*inp
        return out

    def forward(self, inp):
        out = self.bottleneck(inp)
        #out = self.bottleneck2(out)
        return out

class Test2(nn.Module):
    def __init__(self):
        super(Test2, self).__init__()

    def bottleneck(self, inp):
        out = torch.sum(torch.sqrt(inp))
        return out

    def bottleneck2(self, inp):
        out = inp**2 + inp**3 - 2*inp
        return out

    def forward(self, inp, idx):
        out = mycheckpoint(self.bottleneck, inp, idx) # Only difference: use checkpoint here!
        #out = self.bottleneck(inp)
        #out = mycheckpoint(self.bottleneck2, out, idx) # Only difference: use checkpoint here!
        return out

def grad_hess(x,y):
    y.backward(create_graph=True)
    gradient = x.grad.clone().flatten()
    hess = []
    for g in gradient:
        x.grad.zero_()
        g.backward(create_graph=True)
        hess.append(x.grad.clone())
    hessian = torch.stack(hess)
    return gradient, hessian

def weird_hess(x,y):
    y.backward(create_graph=True)
    hess_row = x.grad.clone().flatten()
    x.grad.zero_()
    return hess_row

x1 = torch.tensor([1.0, 2.0], requires_grad=True)
model1 = Test1()
y1 = model1(x1)
g1,h1 = grad_hess(x1,y1)
print(g1)
print(h1)

x2 = torch.tensor([1.0, 2.0], requires_grad=True)
model2 = Test2()
y2 = model2(x2, torch.tensor(0))
h2_0 = weird_hess(x2,y2)
y2 = model2(x2, torch.tensor(1))
h2_1 = weird_hess(x2,y2)
h2 = torch.stack((h2_0,h2_1))
print("Same value:  ",torch.allclose(y1,y2))
print("Same hessian:  ",torch.allclose(h1,h2))

