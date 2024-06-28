# Dataset Processor for CoreScheduler

## Overview

This repository contains Python scripts for processing datasets to be used in large model (LM) training tasks. It is designed to handle large datasets efficiently by using streaming and batching techniques, and saves the processed data in Parquet format, which is compatible with CoreScheduler.

## Tasks

* Text Generation
* TODO ...

## Features

* Multi-Dataset Concatenation: Combines multiple datasets to create a unified training corpus.
* Tokenization: Utilizes transformers.AutoTokenizer for tokenizing text data.
* Batch Processing: Processes data in configurable batch sizes to optimize memory usage.
* Streaming Support: Capable of processing datasets that are too large to fit into memory.
* Data Sharding: Splits the dataset into multiple shards in parquet format, each up to a specified size limit.
* Parquet Output: Saves processed data in Parquet format, making it suitable for integration with systems like CoreScheduler.

## Configuration

Modify the config dictionary in the script to specify:

* Model path
* Data paths and names
* Text column names
* Dataset split ratios and sampling rates

## Usage

Ensure you have the required packages installed:
```bash
conda activate core_scheduler
conda install tqdm datasets transformers -c conda-forge
```

Run the script using:

Modify the "data_path" in config to download and process the dataset.
```bash
cd example/
python dataset/script_name.py
```