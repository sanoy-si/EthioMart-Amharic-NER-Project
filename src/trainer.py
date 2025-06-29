import os
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Sequence
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "data_splits")
MODEL_CHECKPOINT = "xlm-roberta-base"
MODELS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", f"ethio-ner-{MODEL_CHECKPOINT.split('/')[-1]}")

def load_conll_file(file_path):
    """Reads a CoNLL-formatted file and yields examples as dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        tokens, tags = [], []
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[-1]) 
            elif tokens:
                yield {"tokens": tokens, "ner_tags": tags}
                tokens, tags = [], []
        if tokens: 
            yield {"tokens": tokens, "ner_tags": tags}

train_data = list(load_conll_file(os.path.join(DATA_DIR, "train.conll")))
validation_data = list(load_conll_file(os.path.join(DATA_DIR, "validation.conll")))
test_data = list(load_conll_file(os.path.join(DATA_DIR, "test.conll")))

all_tags = set(tag for example in train_data for tag in example["ner_tags"])
label_list = sorted(list(all_tags))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}
num_labels = len(label_list)
print(f"[*] Inferred Labels: {label_list}")

raw_datasets = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(validation_data),
    "test": Dataset.from_list(test_data)
})

ner_feature = Sequence(feature=ClassLabel(names=label_list))
raw_datasets = raw_datasets.cast_column("ner_tags", ner_feature)

print("\n[*] Dataset loaded and parsed successfully:")
print(raw_datasets)

print(f"\n[*] Initializing tokenizer and model for '{MODEL_CHECKPOINT}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label_group in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_group[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("\n[*] Applying tokenization and label alignment...")
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
print("[+] Preprocessing complete.")
print("\nFinal tokenized dataset format:")
print(tokenized_datasets)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=MODELS_OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    
    eval_strategy="steps",
    eval_steps=3,     
    save_strategy="steps",
    save_steps=3,     
    
    logging_steps=10,              
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(f"\n[*] Starting training... Model will be saved to '{MODELS_OUTPUT_DIR}'")
trainer.train()

print("\n[*] Evaluating the best model on the test set...")
test_results = trainer.evaluate(tokenized_datasets["test"])

print("\n--- Test Set Evaluation Results ---")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

print("\n--- Classification Report on Test Set ---")
print(classification_report(true_labels, true_predictions))

BEST_MODEL_DIR = os.path.join(MODELS_OUTPUT_DIR, "best-model")
print(f"\n[+] Saving the best model and tokenizer to '{BEST_MODEL_DIR}'...")
trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

print("\n[***] End-to-end training and evaluation complete! [***]")