# CoreScheduler: A High-Performance Scheduler for Large Model Training

Notice: This project is currently undergoing rapid development. As such, API stability cannot be guaranteed, particularly for dataset APIs. Additionally, several resources, including documentation, are still in the process of being completed. We appreciate your patience as we work to finalize these materials.

If you have any questions, please feel free to open an issue!

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
conda activate core_scheduler
mkdir build
cd build
cmake ..
make -j core_scheduler_tests
./test/core_scheduler_tests
```

Example (see [tutorial](https://docs.google.com/document/d/1cQ0gTcOuIoeZzHIpKj8hwdTGhK9XaAwPAsRrWba1Ijw/edit#heading=h.kjcrsddy5vbj))

```bash
make -j gpt2_single_device
./example/gpt2_single_device
```

```bash
make -j gpt2_ddp
mpirun -np $num_devices$ ./example/gpt2_ddp
```

## Authors

See [AUTHORS.txt](./AUTHORS.txt)
