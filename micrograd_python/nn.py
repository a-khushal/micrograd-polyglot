from micrograd_python.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module): 

class Layer(Module):

class MLP(Module):
