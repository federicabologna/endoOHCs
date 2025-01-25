import os
import time
from collections import defaultdict
import random
import pickle
import pandas as pd
import numpy as np
import random

# For machine learning tools and evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

start_time = time.time()

# Choose the BERT model that we want to use (make sure to keep the cased/uncased consistent)
model_name = 'distilbert-base-uncased'  

# Choose the GPU we want to process this script
device_name = 'cuda'       

# This is the maximum number of tokens in any document sent to BERT
max_length = 512      

endo_dir = '/share/luxlab/roz/endo'

annotation_path = os.path.join(endo_dir, 'intent.csv')
output_path = os.path.join(endo_dir, 'output')

annotations_df = pd.read_csv(annotation_path)
annotations_df = annotations_df[annotations_df["accept"] != "[]"]

# include only label types that had strong accuracy
dir_names = ["SEEKING_EXPERIENCES", "SEEKING_INFO", "SEEKING_EMOTION", "VENT", "PROVIDING_EXPERIENCES"]


for dir_name in dir_names:
  label_start = time.time()
  print(f"RUNNING LABEL {dir_name}")
  # annoying, but there's variation between dir names and our label names
  if dir_name == "SEEKING_EXPERIENCES":
    label_name = "SEEKING EXPERIENCES"

  if dir_name == "SEEKING_INFO":
    label_name = "SEEKING INFORMATIONAL SUPPORT"

  if dir_name == "SEEKING_EMOTION":
    label_name = "SEEKING EMOTIONAL SUPPORT"

  if dir_name == "VENT":
    label_name = "VENT"

  if dir_name == "PROVIDING_EXPERIENCES":
    label_name = "PROVIDING EXPERIENCES"


  label_output_path = os.path.join(output_path, dir_name)
  model_output_path = os.path.join(label_output_path, 'model')

  X = annotations_df.text.tolist()
  y = annotations_df[label_name].tolist()

  train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size = 0.25)
  tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
  train_encodings = tokenizer(train_texts,  truncation=True, padding=True)
  test_encodings = tokenizer(test_texts,  truncation=True, padding=True)

  class SCDataset(torch.utils.data.Dataset):
      def __init__(self, encodings, labels):
          self.encodings = encodings
          self.labels = labels

      def __getitem__(self, idx):
          item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
          item['labels'] = torch.tensor(self.labels[idx])
          return item

      def __len__(self):
          return len(self.labels)

  train_dataset = SCDataset(train_encodings, train_labels)
  test_dataset = SCDataset(test_encodings, test_labels)


  # Set up training arguments
  training_args = TrainingArguments(
      output_dir='./results',          # output directory
      num_train_epochs=3,              # total number of training epochs
      per_device_train_batch_size=16,  # batch size per device during training
      per_device_eval_batch_size=20,   # batch size for evaluation
      learning_rate=5e-5,              # initial learning rate for Adam optimizer
      warmup_steps=50,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
      evaluation_strategy='steps',
  )

  model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")#.to(device_name)

  # Define a custom evaluation function (this could be changes to return accuracy metrics)
  def compute_metrics(pred):
      labels = pred.label_ids
      preds = pred.predictions.argmax(-1)
      acc = accuracy_score(labels, preds)
      return {
          'accuracy': acc,
      }

  trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=test_dataset,            # evaluation dataset
      compute_metrics=compute_metrics      # custom evaluation function
  )

  trainer.train()

  trainer.evaluate()

  # save model
  model.save_pretrained(model_output_path)

  # classification report
  predicted_labels = trainer.predict(test_dataset)
  actual_predicted_labels = predicted_labels.predictions.argmax(-1)
  class_report = classification_report(predicted_labels.label_ids.flatten(), actual_predicted_labels.flatten(), output_dict=True)

  class_report_df = pd.DataFrame(class_report).transpose()
  output_name = 'classification_report_final.csv'
  class_report_df.to_csv(os.path.join(label_output_path, output_name))

  # label end time
  label_end = time.time()

  print(f"LABEL {label_name} FINISHED PROCESSING IN {label_end - label_start} SECONDS.")

# total end time
end_time = time.time()
print(f"ALL LABELS FINISHED PROCESSING IN {end_time - start_time} SECONDS.")

