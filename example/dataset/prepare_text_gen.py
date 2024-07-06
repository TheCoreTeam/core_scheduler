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

import os
import sys
import math
import itertools
from itertools import chain
from collections import deque
import random
from tqdm.auto import tqdm
from multiprocessing import Pool
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer

config = {
    'model_name_or_path': 'openai-community/gpt2',
    'block_size': 1024,
    'data_path': 'dataset_path/',
    'num_proc': 64,
    'seed': 42,
    'process_batch_size': 1000,  # Number of samples to process in each batch
    'streaming': False,  # Set to True if the dataset is too large to fit in memory
    'max_shard_size': int(0.1*1024**3),  # Maximum size of each parquet shard in bytes, default is 0.1GB
    'num_shards': None,  # Number of shards to split the dataset into
    'datasets_info': [
        # {
        #     'dataset_path': 'HuggingFaceFW/fineweb-edu',  # Path to the dataset or a dataset identifier recognized by `load_dataset`
        #     'dataset_name': "sample-10BT",  # If the dataset has multiple configurations or variations
        #     'train_name': 'train',  # Name of the training split
        #     'val_name': None,  # Name of the validation split, if explicitly separate
        #     'test_name': None,  # Name of the test split, if explicitly separate
        #     'text_column': 'text',  # Column name in the dataset that contains text
        #     'remove_columns': ['text','id','dump','url','file_path','language','language_score','token_count','score','int_score'],
        #     'split_ratio': 0.001,  # Optional: ratio to split the training data into validation if no separate validation set is provided
        #     'sampling_rate': 1  # Optional: dataset sampling rate
        # },
        # More datasets can be added in this list
        {
            'dataset_path': 'ivanzhouyq/RedPajama-Tiny',    # For debugging purposes
            'dataset_name': None,
            'train_name': 'train',
            'val_name': None,
            'test_name': None,
            'text_column': 'text',  # Ensure this matches the actual text column name in the dataset
            'remove_columns': ['text', 'meta'],
            'split_ratio': 0.5,
            'sampling_rate': 1
        },
        {
            'dataset_path': 'Trelis/tiny-shakespeare',    # For debugging purposes
            'dataset_name': None,
            'train_name': 'train',
            'val_name': None,
            'test_name': 'test',
            'text_column': 'Text',  # Ensure this matches the actual text column name in the dataset
            'remove_columns': ['Text'],
            'split_ratio': 0.5,
            'sampling_rate': 1
        },
    ]
}

tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], use_fast=True)
if 'Llama' in config['model_name_or_path']:
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
else:
    tokenizer.pad_token = tokenizer.eos_token
tokenize_num_proc = None if tokenizer.is_fast else config['num_proc']

def get_dataset_length(stream_dataset):
    count = 0
    for _ in tqdm(stream_dataset, desc="Counting samples"):
        count += 1
    return count  # To handle cases where the dataset is empty

def sample_stream(dataset, rate):
    n = int(1 / rate)
    for item in itertools.islice(dataset, 0, None, n):
        yield item

def tokenize_function(examples):
    tokenized_output = tokenizer(examples[text_column_name], truncation=True, padding=False, max_length=config['block_size'])
    return {"input_ids": tokenized_output['input_ids']}

def group_texts(examples):
    concatenated_text = sum(examples['input_ids'], [])
    total_length = len(concatenated_text)
    if total_length > config['block_size']:
        total_length = ((total_length - 1) // config['block_size']) * config['block_size']
        input_ids = [concatenated_text[i:i + config['block_size']] for i in range(0, total_length, config['block_size'])]
        labels = [concatenated_text[i + 1:i + 1 + config['block_size']] for i in range(0, total_length, config['block_size'])]
    else:
        # drop the last sentence
        input_ids = []
        labels = []
    return {'input_ids': input_ids, 'labels': labels}

def tokenize_and_group(dataset):
    return dataset.map(tokenize_function, batched=True, num_proc=config['num_proc'], remove_columns=dataset_info['remove_columns']
        ).map(group_texts, batched=True, num_proc=config['num_proc'])

def save_shard(args):
    shard_dataset, output_dir, key, shard_idx, num_shards_calculated = args
    if not isinstance(shard_dataset, Dataset):
        shard_dataset = Dataset.from_dict(shard_dataset)
    shard_path = f'{output_dir}/{key}/data-{shard_idx:05d}-of-{num_shards_calculated:05d}.parquet'
    shard_dataset.to_parquet(shard_path)
    print(f"Shard {shard_idx} saved to {shard_path}")

def getsizeof(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
        Automatically finds the contents of the following builtin containers and
        their subclasses:  tuple, list, deque, dict, set and frozenset.
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    # Add handlers for other types if necessary
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(0)  # estimate sizeof(int) to be 28 bytes
    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)
        if verbose:
            print(s, type(o), repr(o))
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s
    return sizeof(o)

def save_split_as_parquet(dataset, output_dir, max_shard_size=None, num_shards=None, num_proc=1):
    for key, split_dataset in dataset.items():
        print(f"Process subdataset: {key}")
        total_rows = len(split_dataset)
        single_sample_bytes = getsizeof(split_dataset[0])
        total_size_bytes = total_rows * single_sample_bytes
        num_shards_calculated = int((total_size_bytes + max_shard_size - 1) // max_shard_size) \
            if max_shard_size is not None else num_shards if num_shards is not None else 1
        rows_per_shard = (total_rows + num_shards_calculated - 1) // num_shards_calculated
        print(f"The subdataset will be saved in {num_shards_calculated} shards, and each shard has size of {total_size_bytes/1024**3}GB.")
        
        # Prepare arguments for multiprocessing
        args_list = []
        for shard_idx in range(num_shards_calculated):
            start_row = shard_idx * rows_per_shard
            end_row = min((shard_idx + 1) * rows_per_shard, total_rows)
            shard_indices = list(range(start_row, end_row))
            shard_dataset = split_dataset.select(shard_indices)
            args_list.append((shard_dataset, output_dir, key, shard_idx, num_shards_calculated))
        
        # Use multiprocessing to save shards
        effective_num_proc = min(num_proc, num_shards_calculated)
        if effective_num_proc < num_proc:
            print(f"Reduced number of processes from {num_proc} to {effective_num_proc} due to fewer shards than processes.")
        with Pool(effective_num_proc) as pool:
            pool.map(save_shard, args_list)

def process_streaming_dataset(dataset_info):
    """
    Key Logics:
        Validation Dataset Handling:
        If val_name is not provided and split_ratio is None, the validation dataset is set to None.
        If val_name is not provided but split_ratio is provided, the function splits the training dataset into training and validation sets based on the split ratio.
        If val_name is provided, the function directly uses this as the validation dataset.

        Test Dataset Handling:
        If test_name is not provided, the function assigns the validation dataset to the test dataset, essentially using the validation set for testing if no separate test dataset is defined.
        If test_name is provided, the function uses this as the test dataset.

        Error Handling:
        The function checks if train_name is provided. If not, it raises an error.
    """
    try:
        # Load the dataset in streaming mode
        global text_column_name
        text_column_name = dataset_info['text_column']
        dataset = load_dataset(path=dataset_info['dataset_path'], name=dataset_info['dataset_name'], streaming=config['streaming'], num_proc=config['num_proc'])
        train_samples, validation_samples, test_samples = [], [], []

        # Check if train dataset is defined, if not, raise an error
        if dataset_info['train_name'] is None:
            raise ValueError("The dataset must have a 'train_name'.")

        # Process train dataset
        train_stream = dataset[dataset_info['train_name']]
        if dataset_info['val_name'] is None:
            if dataset_info['split_ratio'] is None:
                validation_dataset = None  # No validation data needed
            else:
                # Split train dataset into train and validation datasets based on split_ratio
                for item in tqdm(sample_stream(train_stream, dataset_info['sampling_rate']), desc="Sampling train data"):
                    if random.random() < dataset_info['split_ratio']:
                        validation_samples.append(item[dataset_info['text_column']])
                    else:
                        train_samples.append(item[dataset_info['text_column']])
                validation_dataset = Dataset.from_dict({dataset_info['text_column']: validation_samples})
        else:
            # Use the separate validation dataset
            validation_stream = dataset[dataset_info['val_name']]
            for item in tqdm(validation_stream, desc="Loading validation data"):
                validation_samples.append(item[dataset_info['text_column']])
            validation_dataset = Dataset.from_dict({dataset_info['text_column']: validation_samples})

        # Process test dataset
        if dataset_info['test_name'] is None:
            test_dataset = validation_dataset  # Use validation dataset as test dataset if no separate test dataset is defined
        else:
            test_stream = dataset[dataset_info['test_name']]
            for item in tqdm(test_stream, desc="Loading test data"):
                test_samples.append(item[dataset_info['text_column']])
            test_dataset = Dataset.from_dict({dataset_info['text_column']: test_samples})

        # Convert train samples to dataset
        train_dataset = Dataset.from_dict({dataset_info['text_column']: train_samples})

        # Tokenize and group texts
        print("process train dataset...")
        tokenized_train = tokenize_and_group(train_dataset)
        print("process val dataset...")
        tokenized_val = tokenize_and_group(validation_dataset) if validation_dataset else None
        print("process test dataset...")
        tokenized_test = tokenize_and_group(test_dataset) if test_dataset else None

        return {'train': tokenized_train, 'validation': tokenized_val, 'test': tokenized_test}
    except Exception as e:
        print(f"Failed to process dataset {dataset_info['dataset_path']}: {e}")
        return None

def process_non_streaming_dataset(dataset_info):
    global text_column_name
    text_column_name = dataset_info['text_column']
    dataset = load_dataset(path=dataset_info['dataset_path'], name=dataset_info['dataset_name'], num_proc=config['num_proc'])

    # Sample and split the dataset according to the configuration
    train_dataset = dataset[dataset_info['train_name']]
    sampled_dataset = train_dataset.shuffle(seed=config['seed']).select(
        range(int(len(train_dataset) * dataset_info['sampling_rate'])))
    if dataset_info['val_name'] is None:
        split_index = int(len(sampled_dataset) * dataset_info['split_ratio'])
        train_dataset = sampled_dataset.select(range(split_index, len(sampled_dataset)))
        val_dataset = sampled_dataset.select(range(split_index))
    else:
        train_dataset = sampled_dataset
        val_dataset = dataset[dataset_info['val_name']]

    # Process test dataset
    if dataset_info['test_name'] is None:
        test_dataset = val_dataset  # Use validation dataset as test dataset if no separate test dataset is defined
    else:
        test_dataset = dataset[dataset_info['test_name']]

    # Tokenization and grouping datasets
    print("process train dataset...")
    tokenized_train = tokenize_and_group(train_dataset)
    print("process val dataset...")
    tokenized_val = tokenize_and_group(val_dataset)
    print("process test dataset...")
    tokenized_test = tokenize_and_group(test_dataset)

    return {'train': tokenized_train, 'validation': tokenized_val, 'test': tokenized_test}


if __name__=="__main__":
    print(f"The dataset will be saved in {os.path.abspath(config['data_path'])}")
    tokenized_train_datasets, tokenized_val_datasets, tokenized_test_datasets = [], [], []
    for dataset_info in config['datasets_info']:
        print(f"Processing {dataset_info['dataset_path']}...")
        if config['streaming']:
            processed_datasets = process_streaming_dataset(dataset_info)
        else:
            processed_datasets = process_non_streaming_dataset(dataset_info)
        if processed_datasets:
            tokenized_train_datasets.append(processed_datasets['train'])
            tokenized_val_datasets.append(processed_datasets['validation'])
            tokenized_test_datasets.append(processed_datasets['test'])

    # Concatenate all datasets and shuffle
    tokenized_train_datasets = concatenate_datasets(tokenized_train_datasets).shuffle()
    tokenized_val_datasets = concatenate_datasets(tokenized_val_datasets).shuffle()
    tokenized_test_datasets = concatenate_datasets(tokenized_test_datasets).shuffle()
    final_tokenized_dataset = {
        'train': tokenized_train_datasets,
        'validation': tokenized_val_datasets,
        'test': tokenized_test_datasets
    }

    # Save as parquet format (CoreScheduler format)
    save_split_as_parquet(final_tokenized_dataset, config['data_path'],
                        max_shard_size=config['max_shard_size'], 
                        num_shards=config['num_shards'],
                        num_proc=config['num_proc'])

    # # Save as a single dataset with arrow format (huggingface/datasets format)
    # DatasetDict(final_tokenized_dataset).save_to_disk(
    #                 config['data_path'],
    #                 max_shard_size=config['max_shard_size'], 
    #                 num_shards=config['num_shards'],
    #                 num_proc=config['num_proc'])
