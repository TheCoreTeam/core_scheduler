# CoreScheduler: A High-Performance Scheduler for Large Model Training

## Introduction

https://docs.google.com/document/d/17h5hyaR3e1ARXldxZa8PxdBqlb7a4grDPewjC5P1CyU/edit?usp=sharing

## Usage

Environment:

```bash
git clone git@github.com:TheCoreTeam/core_scheduler.git
cd core_scheduler
conda env create -f env.yaml
```

Compile & Run Tests

```bash
mkdir build
cd build
cmake ..
make -j cs_tests
./test/cs_tests
```
