# CoreScheduler GPT-2 Training Examples

Welcome to the CoreScheduler project! This example is for training the GPT-2 model using our custom C++ framework CoreScheduler. These examples demonstrate the use of single-device and Distributed Data Parallel (DDP) strategies for training. Future versions will expand to more sophisticated distributed strategies like Zero and 3D parallelism.

## Example Contents
* gpt2_single_device.cpp: Training GPT-2 on a single GPU. See [tutorial](https://docs.google.com/document/d/1cQ0gTcOuIoeZzHIpKj8hwdTGhK9XaAwPAsRrWba1Ijw/edit?usp=sharing) for usage.
* gpt2_ddp.cpp: Utilizes Distributed Data Parallel (DDP) for training GPT-2 across multiple GPUs. See [tutorial](https://docs.google.com/document/d/1RKvcaD9XMQ9DE7NFcj0XoSZCvEQXRgA2XDuxaFUPcQY/edit?usp=sharing) for usage.
* (TODO) More distributed strategies are in development.

## Preparation
Before running the examples, ensure that your environment is set up correctly by following the installation instructions on our main [GitHub repository](https://github.com/TheCoreTeam/core_scheduler/), and prepare dataset with [tutorial](../dataset/README.md)

## Future Development
We are actively working on integrating advanced features such as:

* Mixed Precision Training: Current support for fp16 and bf16 types. Future releases will include automatic mixed precision (AMP) support.
* Advanced Distributed Strategies: Planning to introduce Zero and 3D parallelism for efficient scaling across multiple GPUs and nodes.