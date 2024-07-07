# CoreScheduler: A High-Performance Scheduling Framework for Large-Scale Model Training in C++

## Overview

CoreScheduler is a fully-featured C++ library tailored for efficient and scalable training of large models. It excels in managing asynchronous tasks and dependencies in distributed environments. See [introduction](https://docs.google.com/document/d/17h5hyaR3e1ARXldxZa8PxdBqlb7a4grDPewjC5P1CyU/edit?usp=sharing) for more details.

Notice: This project is currently undergoing rapid development. As such, API stability cannot be guaranteed, particularly for dataset APIs. Additionally, several resources, including documentation, are still in the process of being completed. We appreciate your patience as we work to finalize these materials.

If you have any questions, please feel free to open an issue!

## Key Features
- **Pure C++ Implementation**: Optimizes multi-threading and resource management.
- **Asynchronous Scheduling**: Overlaps computation with communication to expedite training.
- **Advanced Scheduling Capabilities**: Enables overlapping of independent GPU computations, significantly enhancing performance.
- **Communication-Computing Overlap**: Efficiently manages data transfer and computation tasks simultaneously to reduce wait times.
- **Computing-Computing Overlap**: Capable of executing multiple computation tasks efficiently, optimizing the use of system capabilities.

## Usage

Clone the repository and set up the environment:

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

## Examples

For detailed tutorials, refer to the examples provided in the example folder. Specific guidance on training the GPT-2 model can be found in the [GPT-2 training tutorial](example/gpt2/README.md).

## Future Directions

* **Enhanced Distributed Strategies**: Future versions will implement advanced strategies like Zero and 3D parallelism to optimize resource allocation and maximize training efficiency across multiple nodes.
* **Distributed Fault Tolerance**: Develop robust fault tolerance mechanisms to ensure consistent training performance and data integrity across distributed systems, even in the event of partial system failures.
* **More Advanced Models (e.g., Llama-3, MoE)**: Expand support for state-of-the-art models including Llama-3 and Mixture of Experts (MoE), enabling cutting-edge research and application in machine learning with enhanced scalability and specialization.

## Contributing

We encourage contributions to CoreScheduler. Please visit our [issues page](https://github.com/TheCoreTeam/core_scheduler/issues) for opportunities to contribute.

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](https://github.com/TheCoreTeam/core_scheduler?tab=Apache-2.0-1-ov-file#) file for details.

## Authors

See [AUTHORS.txt](./AUTHORS.txt)
