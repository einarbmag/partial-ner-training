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
from data_processor import prepare_partial_dataset, convert_partial_labels_to_ignore_index

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

def main():
    # Load dataset
    dataset = load_dataset("wnut_17")
    
    # Initialize tokenizer
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Prepare partially labeled training data (50% of examples will be partially labeled)
    partial_train_dataset = prepare_partial_dataset(
        dataset["train"],
        partial_label_fraction=0.5,
        seed=42
    )
    
    # Tokenize datasets
    tokenized_datasets = {}
    tokenized_datasets["train"] = partial_train_dataset.map(
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
    num_labels = 17  # Number of labels in WNUT dataset
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )
    
    # Initialize data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Initialize trainer
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
    
    # Evaluate the model
    metrics = trainer.evaluate()
    print(f"\nEvaluation metrics:\n{metrics}")

if __name__ == "__main__":
    main()
