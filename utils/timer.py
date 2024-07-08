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