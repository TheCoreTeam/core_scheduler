#pragma once

#define DLLM_SUBMIT_TASK(scheduler, statement) \
  do {                                         \
    auto task = (statement);                   \
    scheduler.submit(std::move(task));         \
  } while (false)
