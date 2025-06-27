import os
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

MODEL_CHECKPOINT = "Davlan/afro-xlm-roberta-base" 
DATA_DIR = "data_splits"
OUTPUT_DIR = f"ethio-ner-{MODEL_CHECKPOINT.split('/')[-1]}"

print(f"[*] Loading raw text datasets from '{DATA_DIR}'...")
data_files = {
    "train": os.path.join(DATA_DIR, "train.conll"),
    "validation": os.path.join(DATA_DIR, "validation.conll"),
    "test": os.path.join(DATA_DIR, "test.conll"),
}
raw_datasets = load_dataset("text", data_files=data_files)
print("\n[*] Text files loaded successfully:", raw_datasets)

def get_label_list(file_path):
    labels = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): # not an empty line
                parts = line.split()
                if len(parts) > 1:
                    labels.add(parts[1])
    return sorted(list(labels))

print("\n[*] Inferring labels from the training data...")
label_list = get_label_list(data_files["train"])
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
num_labels = len(label_list)
print(f"  Label List: {label_list}")
print(f"  Number of Labels: {num_labels}")


print(f"\n[*] Initializing tokenizer and model for '{MODEL_CHECKPOINT}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

def parse_conll_examples(example):
    lines = example['text'].strip().split('\n')
    tokens, tags = [], []
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            tokens.append(parts[0])
            tags.append(label2id[parts[1]]) # Convert tags to IDs
    return {'tokens': tokens, 'ner_tags': tags}

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

print("\n[*] Applying preprocessing to the dataset...")
# Step 1: Parse the CoNLL structure
processed_datasets = raw_datasets.map(parse_conll_examples, remove_columns=['text'])
# Step 2: Tokenize and align the labels
tokenized_datasets = processed_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=processed_datasets["train"].column_names
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
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    metric_for_best_model="f1",
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

print(f"\n[*] Starting training... Model will be saved to '{OUTPUT_DIR}'")
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

BEST_MODEL_DIR = f"{OUTPUT_DIR}/best-model"
print(f"\n[+] Saving the best model and tokenizer to '{BEST_MODEL_DIR}'...")
trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

print("\n[***] End-to-end training and evaluation complete! [***]")