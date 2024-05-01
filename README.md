# DLLM.cpp

## Usage

```bash
conda create -n dllm
conda install cuda-toolkit -c nvidia
conda install gcc gxx openmpi -c conda-forge
```

```bash
git clone git@github.com:TheCoreTeam/DLLM.cpp.git
cd DLLM
git submodule init
git submodule update
mkdir build
cd build
cmake ..
make -j dllm_tests
./test/dllm_tests
```
