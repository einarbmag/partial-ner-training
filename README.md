# Partial NER Training

This repository demonstrates how to train Named Entity Recognition (NER) models on partially labeled data using Hugging Face Transformers. It simulates a realistic scenario where we start with a small fully labeled dataset and incrementally add partially labeled data (e.g., from user feedback on specific entities).

## Overview

The project explores how model performance changes when training with:
1. A fixed small set of fully labeled data (10% of the dataset)
2. Variable amounts of partially labeled data (0%, 10%, 20%, 50%, 90% of the remaining data)
3. Using DistilBERT as the base model for efficient training

Partially labeled data in this context means examples where only some entities are labeled, while others remain unknown. For example:
```python
# Fully labeled example:
{
    'tokens': ['John', 'works', 'at', 'Microsoft', 'in', 'New', 'York'],
    'ner_tags': [1, 0, 0, 3, 0, 5, 5]  # All entities labeled
}

# Partially labeled example (only 'Microsoft' is known):
{
    'tokens': ['John', 'works', 'at', 'Microsoft', 'in', 'New', 'York'],
    'ner_tags': [-1, -1, -1, 3, -1, -1, -1]  # Only Microsoft is labeled
}
```

## Project Structure

```
partial-ner-training/
├── src/
│   ├── data_processor.py    # Data processing utilities
│   ├── train.py            # Training script
│   └── analyze_results.py  # Results analysis script
├── tests/
│   └── test_data_processor.py  # Test suite
└── requirements.txt        # Project dependencies
```

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Run the main training script to execute experiments with different partial label fractions:
```bash
python src/train.py
```

This will:
1. Load the WNUT dataset
2. Select a fixed 10% of data to keep fully labeled
3. Run experiments adding different amounts of partially labeled data
4. Save results for each experiment

### Analyzing Results

View the results across all experiments:
```bash
python src/analyze_results.py
```

This will display a table showing F1 scores, precision, and recall for each partial label fraction:
```
F1 Scores for Different Partial Label Fractions
--------------------------------------------------
  Partial Label % |  F1 Score  | Precision |   Recall
--------------------------------------------------
            0.0% |    0.XXXX  |   0.XXXX  |   0.XXXX
           10.0% |    0.XXXX  |   0.XXXX  |   0.XXXX
           ...
--------------------------------------------------
```

## Implementation Details

### Data Processing
- Uses -1 to mark unknown labels in training data
- Converts -1 to -100 (Hugging Face's ignore index) during training
- Maintains consistent fully labeled subset across experiments
- Randomly selects entities to keep in partial labels

### Model Training
- Uses DistilBERT for efficient training
- Implements custom data collation for mixed label handling
- Tracks metrics across different partial label ratios
- Exports detailed results for analysis

## Testing

Run the test suite:
```bash
python -m pytest tests/test_data_processor.py -v
```

Tests cover:
- Entity span detection
- Partial label creation
- Mixed dataset preparation
- Label conversion utilities

## License

MIT
