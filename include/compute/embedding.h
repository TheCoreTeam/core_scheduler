//
// Created by tingxuan on 2024/5/16.
//
#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::embedding {
TaskCompute forward(const std::shared_ptr<Tensor3D> &output,
                    const std::shared_ptr<const Tensor2D> &input,
                    const std::shared_ptr<const Tensor2D> &wte,
                    const std::shared_ptr<const Tensor2D> &wpe
                    );

TaskCompute backward(const std::shared_ptr<const Tensor3D> &grad_output,
                    const std::shared_ptr<Tensor2D> &grad_input,
                    const std::shared_ptr<Tensor2D> &grad_wte,
                    const std::shared_ptr<Tensor2D> &grad_wpe);

}  // namespace dllm::compute::embedding