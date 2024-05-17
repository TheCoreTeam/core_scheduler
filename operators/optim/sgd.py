import torch
from collections import OrderedDict


class SGD():
    def __init__(self, parameters, lr=1e-3, momentum=0.9):
        if momentum < 0:
            raise ValueError("Momentum should be greater than or equal to 0")
        self.lr = lr
        self.momentum = momentum
        if self.momentum != 0:
            # Initialize velocity for each parameter
            self.velocities = OrderedDict({k: torch.zeros_like(p) for k, p in parameters.items()})

    def step(self, parameters):
        if self.momentum != 0:
            for (k1, p), (k2, v) in zip(parameters.items(), self.velocities.items()):
                v = v.to(p.device)
                # Update velocity
                v.mul_(self.momentum).add_(p.grad, alpha=self.lr)
                # Update parameters
                p.sub_(v)
                parameters[k1] = p
                self.velocities[k2] = v
        else:
            for k, p in parameters.items():
                p.sub_(p.grad, alpha=self.lr)
                parameters[k] = p
        return parameters

