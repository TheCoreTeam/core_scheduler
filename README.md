# DLLM.py

This is a toy project to validate the ideas of DLLM.cpp, good for learning and playing.

## Usage

Environment:
```bash
conda create -n dllm
conda activate dllm
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
please refer to [https://pytorch.org/get-started/locally/] for pytorch installation

Download this project:
```bash
git clone git@github.com:TheCoreTeam/DLLM.cpp.git
cd DLLM.cpp
git checkout dev-dllm.py
```

Run the following commands for operators:
```bash
python test/operators/compute/basic/linear.py
```
Other operators are similar commands

Run the following commands for training GPT2 on single device:
```bash
python test/single_device/train.py
```
Run the following commands for training GPT2 on two devices with data_parallel:
```bash
torchrun --nproc_per_node 2 --nnodes 1  test/data_parallel/train.py
```
Run the following commands for training GPT2 on two devices with zero1:
```bash
torchrun --nproc_per_node 2 --nnodes 1 test/zero/zero1/train.py
```
Run the following commands for training GPT2 on two devices with zero3:
```bash
torchrun --nproc_per_node 2 --nnodes 1 test/zero/zero3/train.py
```
