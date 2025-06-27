import os
from sklearn.model_selection import train_test_split

CONLL_FILE = 'data/labeled/labeled_data.conll'
OUTPUT_DIR = 'data/data_splits'

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CONLL_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

sentences = content.strip().split('\n\n')
print(f"Found {len(sentences)} sentences in the dataset.")

# --- Perform the split ---
# 80% for training, 20% for temp
train_sents, temp_sents = train_test_split(sentences, test_size=0.2, random_state=42)

# 50% of the 20% -> 10% for validation, 10% for test
val_sents, test_sents = train_test_split(temp_sents, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_sents)}")
print(f"Validation set size: {len(val_sents)}")
print(f"Test set size: {len(test_sents)}")

def write_conll_file(filename, sentences_list):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(sentences_list))
        f.write('\n')

write_conll_file(os.path.join(OUTPUT_DIR, 'train.conll'), train_sents)
write_conll_file(os.path.join(OUTPUT_DIR, 'validation.conll'), val_sents)
write_conll_file(os.path.join(OUTPUT_DIR, 'test.conll'), test_sents)

print(f"\n[+] Successfully created data splits in the '{OUTPUT_DIR}' directory.")