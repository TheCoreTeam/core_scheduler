import sys
import torch
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer


config = {
    'model_name_or_path': 'openai-community/gpt2',
    'block_size': 1024,  # Adjust based on the model's maximum input length capabilities
    'data_path': '../data/',  # Directory to save the tokenized datasets
    'datasets_info': [
        {
            'dataset_path': 'karpathy/tiny_shakespeare',  # Path to the dataset or a dataset identifier recognized by `load_dataset`
            'dataset_name': None,  # If the dataset has multiple configurations or variations
            'train_name': 'train',  # Name of the training split
            'val_name': 'validation',  # Name of the validation split, if explicitly separate
            'test_name': 'test',  # Name of the test split, if explicitly separate
            'text_column': 'text',  # Column name in the dataset that contains text
            'split_ratio': 0.1  # Optional: ratio to split the training data into validation if no separate validation set is provided
        }
        # More datasets can be added in this list
    ]
}

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {
        k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size]
            for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def save_split_as_parquet(dataset, output_dir, max_shard_size=None, num_shards=None):
    for key in dataset.keys():
        # Access the specific split
        split_dataset = dataset[key]

        total_rows = len(split_dataset)
        single_sample_bytes = sys.getsizeof(split_dataset[0])
        total_size_bytes = total_rows * single_sample_bytes
        
        # Determine the number of shards
        if max_shard_size is not None:
            # Calculate the number of shards based on the maximum shard size
            num_shards_calculated = (total_size_bytes + max_shard_size - 1) // max_shard_size
            num_shards = num_shards_calculated if num_shards is None else num_shards
        elif num_shards is None:
            # Default to a single shard if no parameters are provided
            num_shards = 1
        rows_per_shard = (total_rows + num_shards - 1) // num_shards

        # Write each shard to a file
        for shard_idx in range(num_shards):
            start_row = shard_idx * rows_per_shard
            end_row = min((shard_idx + 1) * rows_per_shard, total_rows)
            shard_dataset = split_dataset[start_row:end_row]

            # Ensure shard_dataset is a Dataset object
            if not isinstance(shard_dataset, Dataset):
                shard_dataset = Dataset.from_dict(shard_dataset)
            filename = f"data-{shard_idx:05d}-of-{num_shards:05d}.parquet"
            shard_dataset.to_parquet(f'{output_dir}/{key}/{filename}')


tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
if 'Llama' in config['model_name_or_path']:
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
else:
    tokenizer.pad_token = tokenizer.eos_token

block_size = config['block_size']

def tokenize_function(examples):
    return tokenizer(examples[dataset_info['text_column']])

# process multi datasets
tokenized_train_datasets, tokenized_val_datasets, tokenized_test_datasets = [], [], []
for dataset_info in config['datasets_info']:
    print(f"Tokenizing {dataset_info['dataset_path']}...")

    dataset = load_dataset(path=dataset_info['dataset_path'], name=dataset_info['dataset_name'])
    # Split dataset if necessary
    if dataset_info['val_name'] is None:
        if dataset_info['split_ratio'] is not None:
            print(f"Validation set is not provided. Splitting {dataset_info['train_name']} dataset with ratio {dataset_info['split_ratio']}.")
            dataset = dataset[dataset_info['train_name']].train_test_split(test_size=dataset_info['split_ratio'])
            dataset['validation'] = dataset['test']
            del dataset['test']
        else:
            raise ValueError("If val_name is None, the split_ratio must be provided.")
    if dataset_info['test_name'] is None:
        print(f"Test set is not provided. Using {dataset_info['val_name']} as test set.")
        dataset['test'] = dataset['validation'] if dataset_info['val_name'] is None else dataset[dataset_info['val_name']]

    # Rename keys
    new_dataset = DatasetDict({
        'train': dataset[dataset_info['train_name']],
        'validation': dataset[dataset_info['val_name']] if dataset_info['val_name'] is not None else dataset['validation'],
        'test': dataset[dataset_info['test_name']] if dataset_info['test_name'] is not None else dataset['test']
    })

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        num_proc=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        remove_columns=[dataset_info['text_column']],
    )

    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    )
    tokenized_train_datasets.append(tokenized_dataset['train'])
    tokenized_val_datasets.append(tokenized_dataset['validation'])
    tokenized_test_datasets.append(tokenized_dataset['test'])

# save to disk
tokenized_train_datasets = concatenate_datasets(tokenized_train_datasets).shuffle()
tokenized_val_datasets = concatenate_datasets(tokenized_val_datasets).shuffle()
tokenized_test_datasets = concatenate_datasets(tokenized_test_datasets).shuffle()
final_tokenized_dataset = DatasetDict({
    'train': tokenized_train_datasets,
    'validation': tokenized_val_datasets,
    'test': tokenized_test_datasets
})
# final_tokenized_dataset.save_to_disk(config['data_path'])
save_split_as_parquet(final_tokenized_dataset, config['data_path'])
