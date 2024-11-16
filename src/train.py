import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_processor import prepare_mixed_dataset, convert_partial_labels_to_ignore_index

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize all texts and align the labels with them.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten the predictions and labels
    true_predictions = [p for preds in true_predictions for p in preds]
    true_labels = [l for labels in true_labels for l in labels]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='weighted')
    acc = accuracy_score(true_labels, true_predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_and_evaluate(
    dataset,
    tokenizer,
    full_label_indices: np.ndarray,
    partial_label_fraction: float,
    seed: int = 42,
    output_dir: str = "./results"
) -> dict:
    """
    Train and evaluate a model with specific data fractions.
    
    Args:
        dataset: The full dataset
        tokenizer: The tokenizer to use
        full_label_indices: Indices of examples to keep fully labeled
        partial_label_fraction: Fraction of partially labeled data
        seed: Random seed
        output_dir: Directory for outputs
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Prepare the mixed dataset
    mixed_train_dataset = prepare_mixed_dataset(
        dataset["train"],
        full_label_indices=full_label_indices,
        partial_label_fraction=partial_label_fraction,
        seed=seed
    )
    
    # Calculate actual fractions for reporting
    full_label_fraction = len(full_label_indices) / len(dataset["train"])
    
    # Tokenize datasets
    tokenized_datasets = {}
    tokenized_datasets["train"] = mixed_train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_datasets["validation"] = dataset["validation"].map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names
    )
    
    # Initialize model
    model_checkpoint = "distilbert/distilbert-base-cased"
    num_labels = 17  # Number of labels in WNUT dataset
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels
    )
    
    # Define training arguments
    run_name = f"full{full_label_fraction:.2f}_partial{partial_label_fraction:.2f}"
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard logging
    )
    
    # Initialize trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    
    # Add experiment parameters to metrics
    metrics.update({
        "full_label_fraction": full_label_fraction,
        "partial_label_fraction": partial_label_fraction,
        "total_examples": len(mixed_train_dataset),
    })
    
    return metrics

def main():
    # Load dataset and initialize tokenizer
    dataset = load_dataset("wnut_17")
    model_checkpoint = "distilbert/distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Select indices for fully labeled data (10%)
    total_size = len(dataset["train"])
    full_label_size = int(total_size * 0.1)  # 10% fully labeled
    full_label_indices = np.random.permutation(total_size)[:full_label_size]
    
    # Define experiment configurations
    partial_label_fractions = [0.0,0.1, 0.2, 0.5, 0.9]  # Variable partial labeling
    
    # Run experiments
    results = []
    for partial_frac in partial_label_fractions:
        print(f"\nRunning experiment with {partial_frac:.1%} partial labels...")
        metrics = train_and_evaluate(
            dataset=dataset,
            tokenizer=tokenizer,
            full_label_indices=full_label_indices,
            partial_label_fraction=partial_frac,
            output_dir="./results"
        )
        results.append(metrics)
        
        # Print current results
        print(f"\nResults for {partial_frac:.1%} partial labels:")
        print(f"F1 Score: {metrics['eval_f1']:.4f}")
        print(f"Precision: {metrics['eval_precision']:.4f}")
        print(f"Recall: {metrics['eval_recall']:.4f}")
    
    # Save all results
    import json
    with open("./results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
