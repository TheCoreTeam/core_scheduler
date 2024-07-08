# Copyright (c) 2024 The Core Team
#
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch

def load_dataset(dataset_directory):
    # Load the dataset
    dataset = load_from_disk(dataset_directory)
    dataset.set_format(type='torch', columns=['input_ids', 'labels'])

    # Access the splits
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']
    
    return train_dataset, validation_dataset, test_dataset

# Function to convert examples to the format expected by PyTorch
def convert_to_features(example_batch):
    # Convert to PyTorch tensors
    example_batch['input_ids'] = torch.tensor(example_batch['input_ids'])
    example_batch['labels'] = torch.tensor(example_batch['labels'])
    return example_batch

# Define a function to collate data samples into batches
def collate_fn(batch):
    # PyTorch's DataLoader expects a batch to be a tensor, so you need to stack all your batched inputs and labels
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels}
