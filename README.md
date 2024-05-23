# DLLM.cpp

## Usage

Conda packages:
```bash
conda create -n dllm
conda activate dllm
conda install cuda-toolkit -c nvidia
conda install gcc gxx mpich nccl libarrow-dataset -c conda-forge
```

For Ubuntu 20.04 users, you may need to use the system packages:
```bash
```

Run:
```bash
git clone git@github.com:TheCoreTeam/DLLM.cpp.git
cd DLLM.cpp
git submodule init
git submodule update --recursive
mkdir build
cd build
cmake ..
# if you don't want to compile flash attention, use cmake -DDLLM_ENABLE_FLASH_ATTENTION=OFF ..
make -j dllm_tests
./test/dllm_tests
```

## Feature:
1. Task Vector
2. Thread pool
3. Memory release pool
4. Producer-consumer
