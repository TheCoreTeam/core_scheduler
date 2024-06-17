/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "logger.h"

int main(int argc, char **argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  ::testing::InitGoogleTest(&argc, argv);
  const auto error = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return error;
}
