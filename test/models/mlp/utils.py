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

import time
from tqdm.auto import tqdm
import numpy as np
import os
import random
import torch

class MemoryTest:
    """
    A context manager that can be used to measure peak GPU memory usage within a block of code.
    Example:
    ``` python
    with MemoryTest() as mem:
        # Code to measure memory usage
    """
    def __init__(self, enable=True, device='cuda:0', tqdm=False, master_process=True):
        self.enable = enable
        self.device = device
        self.tqdm = tqdm
        self.master_process = master_process

    def __enter__(self):
        if self.enable and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            # torch.cuda.synchronize(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable and torch.cuda.is_available():
            # torch.cuda.synchronize(self.device)
            peak_memory = torch.cuda.max_memory_allocated(self.device)
            self.peak_memory_mb = peak_memory / (1024 * 1024)  # Convert bytes to megabytes
            if self.master_process:
                if self.tqdm:
                    tqdm.write(f"Peak GPU {self.device} Memory Usage: {self.peak_memory_mb:.2f} MB")
                else:
                    print(f"Peak GPU {self.device} Memory Usage: {self.peak_memory_mb:.2f} MB")

class TimeTest:
    """
    A context manager that can be used to measure execution time of a block of code.
    Example:
    ```python
    with TimeTest() as t:
        # Code to measure execution time
    ```
    """
    def __init__(self, enable=True, tqdm=False, master_process=True):
        self.enable = enable
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.tqdm = tqdm
        self.master_process = master_process

    def __enter__(self):
        if self.enable:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.end_time = time.time()
            self.elapsed_time = self.end_time - self.start_time
            if self.master_process:
                if self.tqdm:
                    tqdm.write(f"Execution Time: {self.elapsed_time:.2f} seconds")
                else:
                    print(f"Execution Time: {self.elapsed_time:.2f} seconds")



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


