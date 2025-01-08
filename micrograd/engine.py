import numpy as np

class Value:
    def __init__(self, data, children=()):
        self.data = data
        self.grad = 0.0
        self._backprop = lambda: None
        self._prev = set(children)
    
    def __str__(self):
        return f'{self.data}'
    
    def __repr__(self):
        return f'{self.data}'
    
    def __sub__(self, other):
        out = Value(self.data-other.data, children=(self, other))
        def _backprop():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backprop = _backprop
        return out
       
    
    def __add__(self, other):
        out = Value(self.data+other.data, children=(self, other))
        def _backprop():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backprop = _backprop
        return out

    def __mul__(self, other):
        out = Value(self.data*other.data, children=(self, other))
        def _backprop():
            self.grad = other.data*out.grad
            other.grad = self.data*out.grad
        out._backprop = _backprop
        return out

    def relu(self):
        if self.data < 0:
            out = Value(0.0)
        else:
            out = Value(self.data, children=(self,))
        
        def _backprop():
            if self.data < 0:
                self.grad += 0
            else:
                self.grad += out.grad
        out._backprop = _backprop
        return out
    
    def backprop(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
            topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backprop()


if __name__ == '__main__':
    a = Value(1.0)
    b = Value(2.0)
    c = a*b
    d = c.relu()

    d.backprop()
    print(a.grad)
    print(b.grad)