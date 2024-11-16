import pytest
from datasets import Dataset
import random
import numpy as np
from src.data_processor import (
    find_entity_spans,
    create_partial_labels,
    prepare_partial_dataset,
    convert_partial_labels_to_ignore_index
)

@pytest.fixture
def sample_sequence():
    """Fixture providing a sample sequence with entities."""
    return {
        'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.'],
        'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0]
    }

@pytest.fixture
def complex_sequence():
    """Fixture providing a more complex sequence with multiple entity types."""
    return {
        'tokens': ['John', 'works', 'at', 'Microsoft', 'in', 'New', 'York', 'City'],
        'ner_tags': [1, 0, 0, 3, 0, 5, 6, 6]  # 1=PERSON, 3=ORG, 5,6=LOCATION
    }

def test_find_entity_spans_basic(sample_sequence):
    """Test basic entity span detection."""
    spans = find_entity_spans(sample_sequence['tokens'], sample_sequence['ner_tags'])
    assert len(spans) == 2
    # Check Empire State Building span
    assert spans[0] == (14, 17, 7)
    # Check ESB span
    assert spans[1] == (18, 19, 7)

def test_find_entity_spans_complex(complex_sequence):
    """Test entity span detection with multiple entity types."""
    spans = find_entity_spans(complex_sequence['tokens'], complex_sequence['ner_tags'])
    assert len(spans) == 3
    # Check spans for each entity
    assert spans[0] == (0, 1, 1)  # John
    assert spans[1] == (3, 4, 3)  # Microsoft
    assert spans[2] == (5, 8, 5)  # New York City

def test_find_entity_spans_empty():
    """Test entity span detection with no entities."""
    tokens = ['This', 'is', 'a', 'test', '.']
    ner_tags = [0, 0, 0, 0, 0]
    spans = find_entity_spans(tokens, ner_tags)
    assert len(spans) == 0

def test_create_partial_labels_keep_one(sample_sequence):
    """Test partial label creation with keep_prob=1.0."""
    random.seed(42)  # For reproducibility
    result = create_partial_labels(sample_sequence, keep_prob=1.0)
    
    # Count non-masked labels
    non_masked = sum(1 for tag in result['ner_tags'] if tag != -1)
    
    # Should have exactly one entity's worth of labels
    assert non_masked > 0
    # All other labels should be masked
    assert sum(1 for tag in result['ner_tags'] if tag == -1) > 0
    # Original tokens should be unchanged
    assert result['tokens'] == sample_sequence['tokens']

def test_create_partial_labels_keep_none(sample_sequence):
    """Test partial label creation with keep_prob=0.0."""
    result = create_partial_labels(sample_sequence, keep_prob=0.0)
    # Should be identical to input
    assert result == sample_sequence

def test_create_partial_labels_no_entities():
    """Test partial label creation with a sequence containing no entities."""
    example = {
        'tokens': ['This', 'is', 'a', 'test', '.'],
        'ner_tags': [0, 0, 0, 0, 0]
    }
    result = create_partial_labels(example, keep_prob=1.0)
    # Should be identical to input since there are no entities to mask
    assert result == example

def test_prepare_partial_dataset():
    """Test dataset preparation with partial labels."""
    # Create a small test dataset
    data = {
        'tokens': [
            ['John', 'works', 'at', 'Microsoft'],
            ['Visit', 'New', 'York'],
            ['Hello', 'world', '.']
        ],
        'ner_tags': [
            [1, 0, 0, 3],
            [0, 5, 6],
            [0, 0, 0]
        ]
    }
    dataset = Dataset.from_dict(data)
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Prepare dataset with 50% partial labels
    result = prepare_partial_dataset(dataset, partial_label_fraction=0.5, seed=42)
    
    # Check that the dataset has the same length
    assert len(result) == len(dataset)
    
    # Check that some examples have been partially labeled
    has_partial = False
    has_full = False
    for item in result:
        if -1 in item['ner_tags']:
            has_partial = True
        else:
            has_full = True
    
    assert has_partial and has_full

def test_convert_partial_labels_to_ignore_index():
    """Test conversion of partial labels to ignore index."""
    labels = [1, -1, 2, -1, 0, 3, -1]
    result = convert_partial_labels_to_ignore_index(labels)
    
    expected = [1, -100, 2, -100, 0, 3, -100]
    assert result == expected

def test_convert_partial_labels_to_ignore_index_custom():
    """Test conversion of partial labels to custom ignore index."""
    labels = [1, -1, 2, -1, 0, 3, -1]
    result = convert_partial_labels_to_ignore_index(labels, ignore_index=-999)
    
    expected = [1, -999, 2, -999, 0, 3, -999]
    assert result == expected
