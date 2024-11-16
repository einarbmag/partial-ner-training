# Partial NER Training

This repository demonstrates how to train Named Entity Recognition (NER) models on partially labeled data using Hugging Face Transformers. It shows how to effectively utilize both fully labeled and partially labeled data in the training process.

## Overview

In real-world scenarios, we often have datasets where only some entities are labeled, while others remain unlabeled. This project shows how to handle such scenarios by:

1. Converting fully labeled data into partially labeled data for demonstration
2. Using special token values (-1) to mark unknown labels
3. Converting unknown labels to the model's ignore index (-100) during training
4. Training a BERT-based model on the mixed dataset

## Project Structure

```
partial-ner-training/
├── src/
│   ├── data_processor.py    # Data processing utilities
│   └── train.py            # Training script
└── requirements.txt        # Project dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The main training script can be run with:

```bash
python src/train.py
```

This will:
1. Load the WNUT dataset
2. Convert 50% of the training examples to partial labels
3. Train a BERT model on the mixed dataset
4. Evaluate the model on the validation set

## Data Processing

The `data_processor.py` module provides several key functions:

- `find_entity_spans`: Identifies entity spans in the input sequence
- `create_partial_labels`: Converts fully labeled examples to partially labeled ones
- `prepare_partial_dataset`: Prepares a dataset with a mix of fully and partially labeled examples
- `convert_partial_labels_to_ignore_index`: Converts partial labels (-1) to the model's ignore index (-100)

## Example

For instance, given the input:
```python
{
    'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.'],
    'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0]
}
```

The partial labeling might keep only the "Empire State Building" entity and mask all other labels:
```python
{
    'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.'],
    'ner_tags': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 8, 8, -1, -1, -1]
}
```

## License

MIT
