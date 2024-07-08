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

from tqdm.auto import tqdm
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
            peak_memory = torch.cuda.max_memory_allocated(self.device)
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