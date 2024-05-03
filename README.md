# DLLM.cpp

## Usage

Conda packages:
```bash
conda create -n dllm
conda activate dllm
conda install cuda-toolkit -c nvidia
conda install gcc=13 gxx=13 mpich nccl -c conda-forge
```

For Ubuntu 20.04 users, you may need to use the system packages:
```bash
```

Run:
```bash
git clone git@github.com:TheCoreTeam/DLLM.cpp.git
cd DLLM.cpp
git submodule init
git submodule update
mkdir build
cd build
cmake ..
make -j dllm_tests
./test/dllm_tests
```

## Feature:
1. Task Vector
2. Thread pool
3. Memory release pool
