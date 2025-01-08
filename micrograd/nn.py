from engine import Value
import random

class Neuro():
    def __init__(self, din, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(din)]
        self.b = Value(0)
        self.nonlin = nonlin
    
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return self.w+[self.b]

class Layer():
    def __init__(self, din, dout, **kwargs):
        self.neuros = [Neuro(din, **kwargs) for _ in range(dout)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neuros]
        return out if len(out) > 1 else out[0]
    
    def parameters(self):
        return [p for n in self.neuros for p in n.parameters()]

class MLP():
    def __init__(self, din, douts):
        sz = [din] + douts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(douts)-1) for i in range(len(douts))]
    
    def __call__(self, x):
        t = x
        for layer in self.layers:
            t = layer(t)
        return t
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    
    
if __name__ == '__main__':
    m = MLP(2, [4, 4, 1])
    X = [
        [Value(0),Value(0)], 
        [Value(1),Value(0)],
        [Value(0),Value(1)], 
        [Value(1),Value(1)]
        ]
    y = [1,0,0,1]

    pred = m(X[3])
    parameters = m.parameters()
    # print(pred)
    # print(parameters)

    epoch = 2000
    for i in range(epoch):
        pred_y = [m(x) for x in X]

        loss = Value(0)
        for i in range(len(y)):
            loss += (pred_y[i]-Value(y[i])) * (pred_y[i]-Value(y[i]))
        loss.backprop()
        print(loss)

        step = 0.05
        params = m.parameters()
        for p in parameters:
            p.data -= step*p.grad
            p.grad = 0
        
                
 
    