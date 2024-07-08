# Copyright (c) 2024 The Core Team
#
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from collections import OrderedDict


class SGD():
    def __init__(self, parameters, lr=1e-3, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False, maximize=False):
        if momentum < 0 or dampening < 0 or weight_decay < 0:
            raise ValueError("Momentum, dampening, and weight decay should be non-negative")
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        if self.momentum != 0:
            # Initialize velocity for each parameter
            self.velocities = OrderedDict({k: torch.zeros_like(p, device=p.device) for k, p in parameters.items()})

    def step(self, parameters):
        for k, p in parameters.items():
            if p.grad is None:
                continue
            d_p = p.grad
            if self.weight_decay != 0:
                d_p = d_p.add(p, alpha=self.weight_decay)
            
            if self.maximize:
                d_p = -d_p

            if self.momentum != 0:
                v = self.velocities[k]
                # Update velocity
                v.mul_(self.momentum).add_(d_p, alpha=1 - self.dampening)

                if self.nesterov:
                    # Nesterov update: temporarily adjust the direction
                    d_p = d_p.add(v, alpha=self.momentum)
                else:
                    d_p = v

            # Update parameters
            p.add_(d_p, alpha=-self.lr)

        return parameters
    
    def zero_grad(self, parameters):
        for param in parameters.values():
            if param.grad is not None:
                param.grad.zero_()
