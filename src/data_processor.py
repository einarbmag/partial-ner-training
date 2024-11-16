import random
from typing import List, Dict, Any, Tuple
import numpy as np
from datasets import Dataset
from collections import defaultdict

def find_entity_spans(tokens: List[str], ner_tags: List[int]) -> List[Tuple[int, int, int]]:
    """
    Find spans of entities in the sequence.
    Returns list of tuples (start_idx, end_idx, entity_tag)
    """
    spans = []
    current_entity = None
    start_idx = None
    
    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag != 0:  # Part of an entity
            if current_entity is None:  # Start of new entity
                current_entity = tag
                start_idx = i
            elif tag != current_entity and tag % 2 == 0:  # Continue previous entity
                continue
            elif tag != current_entity:  # Start of new entity
                if start_idx is not None:
                    spans.append((start_idx, i, current_entity))
                current_entity = tag
                start_idx = i
        else:  # Outside entity
            if current_entity is not None:
                spans.append((start_idx, i, current_entity))
                current_entity = None
                start_idx = None
    
    # Handle entity at end of sequence
    if current_entity is not None:
        spans.append((start_idx, len(tokens), current_entity))
    
    return spans

def create_partial_labels(example: Dict[str, Any], keep_prob: float = 1.0) -> Dict[str, Any]:
    """
    Transform fully labeled data into partially labeled data by randomly keeping
    some entities and masking others with -1.
    
    Args:
        example: Dictionary containing 'tokens' and 'ner_tags'
        keep_prob: Probability of keeping the example as partially labeled (vs fully labeled)
    
    Returns:
        Dictionary with modified 'ner_tags' where some labels are masked with -1
    """
    # Copy the example to avoid modifying the original
    new_example = dict(example)
    
    # Randomly decide whether to partially label this example
    if random.random() >= keep_prob:
        return new_example
    
    tokens = example['tokens']
    ner_tags = example['ner_tags']
    
    # Find all entity spans
    spans = find_entity_spans(tokens, ner_tags)
    
    if not spans:  # No entities in this example
        return new_example
    
    # Randomly select one entity span to keep
    kept_span = random.choice(spans)
    
    # Create new tags with all -1 except for the kept entity
    new_tags = [-1] * len(ner_tags)
    start_idx, end_idx, entity_tag = kept_span
    
    # Restore the kept entity's tags
    for i in range(start_idx, end_idx):
        new_tags[i] = ner_tags[i]
    
    new_example['ner_tags'] = new_tags
    return new_example

def prepare_partial_dataset(
    dataset: Dataset,
    partial_label_fraction: float = 0.5,
    seed: int = 42
) -> Dataset:
    """
    Prepare a dataset with a mix of fully and partially labeled examples.
    
    Args:
        dataset: The input dataset
        partial_label_fraction: Fraction of examples to convert to partial labels
        seed: Random seed for reproducibility
    
    Returns:
        Dataset with mixed fully and partially labeled examples
    """
    random.seed(seed)
    
    # Apply partial labeling to the specified fraction of the dataset
    return dataset.map(
        lambda x: create_partial_labels(x, keep_prob=partial_label_fraction),
        load_from_cache_file=False
    )

def convert_partial_labels_to_ignore_index(
    labels: List[int],
    ignore_index: int = -100
) -> List[int]:
    """
    Convert partial labels (-1) to the model's ignore index (-100)
    """
    return [ignore_index if label == -1 else label for label in labels]

def prepare_mixed_dataset(
    dataset: Dataset,
    full_label_indices: np.ndarray,
    partial_label_fraction: float = 0.2,
    seed: int = 42
) -> Dataset:
    """
    Prepare a dataset with a fixed set of fully labeled data and an additional
    fraction of partially labeled data.
    
    Args:
        dataset: The input dataset
        full_label_indices: Indices of examples to keep fully labeled
        partial_label_fraction: Fraction of remaining data to add as partially labeled
        seed: Random seed for reproducibility
    
    Returns:
        Dataset with mixed fully and partially labeled examples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total_size = len(dataset)
    
    # Get indices not in full_label_indices for partial labeling
    all_indices = set(range(total_size))
    full_indices_set = set(full_label_indices)
    remaining_indices = list(all_indices - full_indices_set)
    
    # Randomly select indices for partial labeling
    partial_label_size = int(len(remaining_indices) * partial_label_fraction)
    partial_indices = set(np.random.choice(
        remaining_indices, 
        size=partial_label_size, 
        replace=False
    ))
    
    # Create the mixed dataset
    def process_example(example, idx):
        if idx in full_indices_set:
            return example
        elif idx in partial_indices:
            return create_partial_labels(example, keep_prob=1.0)
        else:
            return None
    
    # Apply the processing and filter out None results
    processed_examples = []
    for idx, example in enumerate(dataset):
        processed = process_example(example, idx)
        if processed is not None:
            processed_examples.append(processed)
    
    # Convert back to Dataset format
    return Dataset.from_list(processed_examples)
