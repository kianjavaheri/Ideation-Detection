import pandas as pd

csv_path = "mapped_dataset.csv"
df = pd.read_csv(csv_path)

from sklearn.model_selection import train_test_split

# Split the data
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['class'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)

# Stratify chooses what's being split
# Random-state ensures the state is maintaied (doesn't have to be 42)











from transformers import DistilBertTokenizerFast

# Tokenize the data
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
# This is the tokenizer for DistilBERT

train_encodings = tokenizer(
    train_df['text'].tolist(),    # 1. Raw list of text examples
    padding=True,                 # 2. Pad all sequences to the same length
    truncation=True,              # 3. Cut off sequences that are too long (DistilBERT max = 512 tokens)
    return_tensors="pt"           # 4. Return PyTorch tensors (instead of plain lists or NumPy arrays)
)

val_encodings = tokenizer(
    val_df['text'].tolist(),
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print("Example input IDs:", train_encodings['input_ids'][0])
# 101 represents the start of a sequence [CLS]
# The larger numbers represent the tokens in the vocabulary
# 102 represents the end of a sequence [SEP]
# 0 represents the padding token

print("Attention mask:", train_encodings['attention_mask'][0])
# The attention mask indicates which tokens are padding (0) and which are not (1)

print("Decoded text:", tokenizer.decode(train_encodings['input_ids'][0]))
# The decoded text should match the original text in the dataset

# NOTE: We see that everything is as expected so far












import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset): # Inherits from PyTorch's Dataset class
    def __init__(self, encodings, labels):
        self.encodings = encodings # gives input IDs and attention masks
        self.labels = labels # gives the labels (0 or 1)

    def __len__(self): # Returns the number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx): # Returns a single sample from the dataset
        # Grab the inputs for a single sample
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_labels = train_df['class'].tolist()
val_labels = val_df['class'].tolist()

train_dataset = TextClassificationDataset(train_encodings, train_labels)
val_dataset = TextClassificationDataset(val_encodings, val_labels)

# Check dataset sizes
print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))

# Pulls the first sample from the training dataset
# Returns a dictionary with input IDs, attention mask, and labels
sample = train_dataset[0]

# Keys you should see: input_ids, attention_mask, labels
print("Keys in sample:", sample.keys())

# Print the values of the sample
print("input_ids:", sample['input_ids'])
print("attention_mask:", sample['attention_mask'])
print("label:", sample['labels'])

# Decode the input IDs back to text to confirm
decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
print("Decoded text:", decoded)