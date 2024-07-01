import pandas as pd
import nlpaug.augmenter.word as naw
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
data_path = os.path.join(ROOT_DIR,"data")

model_name= "distilbert-base-uncased"

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Columns to read
columns_to_read = ["Description", "label"]

# Read the data
data = pd.read_csv(os.path.join(data_path,"categorized_transactions.csv"),usecols=columns_to_read)

# Display the first few rows
print(data.head())

data.shape[0]


# Initialize the synonym augmentation
syn_aug = naw.SynonymAug(aug_src='wordnet')

# Function to apply augmentation
def augment_text(text, augmenter):
    augmented_text = augmenter.augment(text)
    return augmented_text

# Apply augmentation to the "Description" column in the training data
data['augmented_Description'] = data['Description'].apply(lambda x: augment_text(x, syn_aug))

# Combine original and augmented descriptions
augmented_data = pd.DataFrame({
    'Description': data['augmented_Description'],
    'label': data['label']
})

# Combine the original training data with the augmented data
#data_augmented = pd.concat([train_data[['Description', 'label']], augmented_data])

# Shuffle the augmented training data
train_data_augmented = augmented_data.sample(frac=1).reset_index(drop=True)

# Display the first few rows of the augmented training data
print(train_data_augmented.head(20))

train_data_augmented.shape[0]

labels = train_data_augmented["label"].unique().tolist()
labels =  [s.strip().lower() for s in labels]
#labels = [s.lower() for s in labels]

NUM_LABELS = len(labels)
id2label = {i: label for i, label in enumerate(labels)}
label2id= {label: i for i,label in enumerate(labels)}

train_data_augmented["label_id"] =train_data_augmented["label"].map(lambda x: label2id[x.lower()])

SIZE = train_data_augmented.shape[0]
train_texts = list(train_data_augmented["Description"][:SIZE//2]) #all until half of the size of the data
val_texts = list(train_data_augmented["Description"][SIZE//2: (3*SIZE//4)]) # from half of the data until 3/4
test_texts = list(train_data_augmented["Description"][(3*SIZE//4):]) # from 3/4 of the data
train_labels= list(train_data_augmented["label_id"][:SIZE//2])
val_labels= list(train_data_augmented["label_id"][SIZE//2: (3*SIZE//4)])
test_labels= list(train_data_augmented["label_id"][(3*SIZE//4):])

train_texts =[item for sublist in train_texts for item in sublist]
val_texts =[item for sublist in val_texts for item in sublist]
test_texts =[item for sublist in test_texts for item in sublist]

tokenizer= AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS,
                                                           ignore_mismatched_sizes=True, id2label=id2label, label2id=label2id)
model.to(device)

train_enconding = tokenizer(train_texts, truncation=True, padding=True)
val_enconding = tokenizer(val_texts, truncation=True, padding=True)
test_enconding = tokenizer(test_texts, truncation=True, padding=True)

import torch
from torch.utils.data import Dataset
class DataLoader(Dataset):
  def __init__(self, enconding, labels):
    self.enconding = enconding
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.enconding.items()}
    item["labels"] = torch.tensor(self.labels[idx])
    return item

#loads the encoded train texts and their labels
train_dataloader = DataLoader(train_enconding, train_labels)
val_dataloader = DataLoader(val_enconding, val_labels)
test_dataloader = DataLoader(test_enconding, test_labels)

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Setup evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
      "accuracy": accuracy_score(labels, predictions),
      "f1": f1_score(labels, predictions, average="weighted"),
      "precision": precision_recall_fscore_support(labels, predictions, average="weighted")[0],
      "recall": precision_recall_fscore_support(labels, predictions, average="weighted")[1]
  }

training_args = TrainingArguments(
    output_dir="../src/results",
#The number of epochs
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,

#Number of steps used for linear warm up
    warmup_steps=50,
    weight_decay=0.01,

#Tensorboard log directory
    logging_dir='../src/logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    gradient_accumulation_steps=4,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    compute_metrics=compute_metrics,
)

trainer.train()

model_path = os.path.join(ROOT_DIR, "model", "kenyan_bank_transaction_model_2")
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

from transformers import pipeline

model_path= os.path.join(ROOT_DIR, "model", "bank_text_classification")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

