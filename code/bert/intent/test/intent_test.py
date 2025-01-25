import os
import json
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
raw_output_path = os.path.join(output_path, 'predictions_raw_data.json')

annotations_df = pd.read_csv(annotation_path)
annotations_df = annotations_df[annotations_df["accept"] != "[]"]

def get_sent_count(cell_text):
  sentences = sent_tokenize(cell_text)
  sent_num = len(sentences)
  return sent_num

def get_word_count(cell_text):
  words = word_tokenize(cell_text)
  word_num = len(words)
  return word_num

def get_sentence_segments(cell_text, sections = "f_and_l", n = 10):
  sentences = sent_tokenize(cell_text)
  num_sent = len(sentences)

  # if getting first and last
  if sections == "f_and_l":
    if num_sent > n:
      split_n = int(n/2)
      first = sentences[:split_n]
      last = sentences[-split_n:]
      combined_sents = first + last
      combined_sents = " ".join(combined_sents)
    else:
      combined_sents = cell_text

  # if getting first
  elif sections == "f":
    if num_sent > n:
      combined_sents = sentences[0:n]
      combined_sents = " ".join(combined_sents)
    else:
      combined_sents = cell_text

  # if getting last
  elif sections == "l":
    if num_sent > n:
      combined_sents = sentences[-n:]
      combined_sents = " ".join(combined_sents)
    else:
      combined_sents = cell_text

  # if getting middle
  elif sections == "m":
    if num_sent > n:
      middle_index = round(len(sentences)/2)
      half_n = n/2
      first_pos = middle_index - half_n
      last_pos = middle_index + half_n
      combined_sents = sentences[int(first_pos):int(last_pos)]
      combined_sents = " ".join(combined_sents)
    else:
      combined_sents = cell_text

  # if getting random
  elif sections == "r":
    if num_sent > n:
      combined_sents = random.sample(sentences,n)
      combined_sents = " ".join(combined_sents)
    else:
      combined_sents = cell_text

  elif sections == "all":
    combined_sents = cell_text

  else:
    combined_sents = cell_text

  return combined_sents


annotations_df["num_sents"] = annotations_df["text"].apply(get_sent_count)
annotations_df["num_words"] = annotations_df["text"].apply(get_word_count)


dir_names = ["SEEKING_EXPERIENCES", "SEEKING_INFO", "SEEKING_EMOTION", "VENT", "PROVIDING_EXPERIENCES"]

predictions_main_dict = []

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

  if dir_name == "PROVIDING_INFO":
    label_name = "PROVIDING INFORMATIONAL SUPPORT"

  if dir_name == "PROVIDING_EMOTION":
    label_name = "PROVIDING EMOTIONAL SUPPORT"

  label_output_path = os.path.join(output_path, dir_name)

  # WITH CROSS VAL LOOP
  seeds = [0, 5, 30, 42]
  segments = [4, 8, 10, 12, 16]
  parts = ["f", "f_and_l", "l", "m"]

  for part in parts:
    print(f"RUNNING PART {part}")
    for num in segments:
      print(f"RUNNING SEGMENT {num}")
      
      zero_precision = []
      zero_recall = []
      zero_f1 = []
      zero_support = []

      one_precision = []
      one_recall = []
      one_f1 = []
      one_support = []

      acc_precision = []
      acc_recall = []
      acc_f1 = []
      acc_support = []

      macro_precision = []
      macro_recall = []
      macro_f1 = []
      macro_support = []

      weighted_precision = []
      weighted_recall = []
      weighted_f1 = []
      weighted_support = []
    
      for seed in seeds:

        test_df = annotations_df
        test_df["shortened_post"] = test_df["text"].apply(get_sentence_segments, sections = part, n = num)

        X = test_df.shortened_post.tolist()
        y = test_df[label_name].tolist()

        train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = seed)
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


        predicted_labels = trainer.predict(test_dataset)
        actual_predicted_labels = predicted_labels.predictions.argmax(-1)
        class_report = classification_report(predicted_labels.label_ids.flatten(), actual_predicted_labels.flatten(), output_dict=True)

        predictions_for_dict = predicted_labels.label_ids.flatten()

        # create dictionary
        predictions_dict = {"label": label_name, "length": num, "section": part, "seed": seed, "predictions": predictions_for_dict.tolist()}

        predictions_main_dict.append(predictions_dict)

        zero_precision.append(class_report['0']['precision'])
        zero_recall.append(class_report['0']['recall'])
        zero_f1.append(class_report['0']['f1-score'])
        zero_support.append(class_report['0']['support'])

        one_precision.append(class_report['1']['precision'])
        one_recall.append(class_report['1']['recall'])
        one_f1.append(class_report['1']['f1-score'])
        one_support.append(class_report['1']['support'])

        macro_precision.append(class_report['macro avg']['precision'])
        macro_recall.append(class_report['macro avg']['recall'])
        macro_f1.append(class_report['macro avg']['f1-score'])
        macro_support.append(class_report['macro avg']['support'])

        weighted_precision.append(class_report['weighted avg']['precision'])
        weighted_recall.append(class_report['weighted avg']['recall'])
        weighted_f1.append(class_report['weighted avg']['f1-score'])
        weighted_support.append(class_report['weighted avg']['support'])

        print(f"DONE WITH SEEDS")

      zero_precision_score = sum(zero_precision) / len(zero_precision)
      zero_recall_score = sum(zero_recall) / len(zero_recall)
      zero_f1_score = sum(zero_f1) / len(zero_f1)
      zero_support_score = sum(zero_support) / len(zero_support)

      one_precision_score = sum(one_precision) / len(one_precision)
      one_recall_score = sum(one_recall) / len(one_recall)
      one_f1_score = sum(one_f1) / len(one_f1)
      one_support_score = sum(one_support) / len(one_support)

      macro_precision_score = sum(macro_precision) / len(macro_precision)
      macro_recall_score = sum(macro_recall) / len(macro_recall)
      macro_f1_score = sum(macro_f1) / len(macro_f1)
      macro_support_score = sum(macro_support) / len(macro_support)

      weighted_precision_score = sum(weighted_precision) / len(weighted_precision)
      weighted_recall_score = sum(weighted_recall) / len(weighted_recall)
      weighted_f1_score = sum(weighted_f1) / len(weighted_f1)
      weighted_support_score = sum(weighted_support) / len(weighted_support)

      zero_dict = {'precision': zero_precision_score, 'recall': zero_recall_score, 'f1-score': zero_f1_score, 'support': zero_support_score}
      one_dict = {'precision': one_precision_score, 'recall': one_recall_score, 'f1-score': one_f1_score, 'support': one_support_score}
      macro_dict = {'precision': macro_precision_score, 'recall': macro_recall_score, 'f1-score': macro_f1_score, 'support': macro_support_score}
      weighted_dict = {'precision': weighted_precision_score, 'recall': weighted_recall_score, 'f1-score': weighted_f1_score, 'support': weighted_support_score}

      combined_dict = {'0': zero_dict, '1': one_dict, 'macro avg': macro_dict, 'weighted avg': weighted_dict}

      class_report_df = pd.DataFrame(combined_dict).transpose()
      output_name = 'classification_report_' + part + '_' + str(num) + '_'+ '.csv'
      class_report_df.to_csv(os.path.join(label_output_path, output_name))


  # get overall accuracy
  print(f"RUNNING OVERALL ACCURACY")
  zero_precision = []
  zero_recall = []
  zero_f1 = []
  zero_support = []

  one_precision = []
  one_recall = []
  one_f1 = []
  one_support = []

  acc_precision = []
  acc_recall = []
  acc_f1 = []
  acc_support = []

  macro_precision = []
  macro_recall = []
  macro_f1 = []
  macro_support = []

  weighted_precision = []
  weighted_recall = []
  weighted_f1 = []
  weighted_support = []

  for seed in seeds:

    X = annotations_df.text.tolist()
    y = annotations_df[label_name].tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = seed)
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

    predicted_labels = trainer.predict(test_dataset)
    actual_predicted_labels = predicted_labels.predictions.argmax(-1)
    class_report = classification_report(predicted_labels.label_ids.flatten(), actual_predicted_labels.flatten(), output_dict=True)
  
    predictions_for_dict = predicted_labels.label_ids.flatten()

    # create dictionary
    predictions_dict = {"label": label_name, "length": "all", "section": "all", "seed": seed, "predictions": predictions_for_dict.tolist()}

    predictions_main_dict.append(predictions_dict)

    zero_precision.append(class_report['0']['precision'])
    zero_recall.append(class_report['0']['recall'])
    zero_f1.append(class_report['0']['f1-score'])
    zero_support.append(class_report['0']['support'])

    one_precision.append(class_report['1']['precision'])
    one_recall.append(class_report['1']['recall'])
    one_f1.append(class_report['1']['f1-score'])
    one_support.append(class_report['1']['support'])

    macro_precision.append(class_report['macro avg']['precision'])
    macro_recall.append(class_report['macro avg']['recall'])
    macro_f1.append(class_report['macro avg']['f1-score'])
    macro_support.append(class_report['macro avg']['support'])

    weighted_precision.append(class_report['weighted avg']['precision'])
    weighted_recall.append(class_report['weighted avg']['recall'])
    weighted_f1.append(class_report['weighted avg']['f1-score'])
    weighted_support.append(class_report['weighted avg']['support'])

  zero_precision_score = sum(zero_precision) / len(zero_precision)
  zero_recall_score = sum(zero_recall) / len(zero_recall)
  zero_f1_score = sum(zero_f1) / len(zero_f1)
  zero_support_score = sum(zero_support) / len(zero_support)

  one_precision_score = sum(one_precision) / len(one_precision)
  one_recall_score = sum(one_recall) / len(one_recall)
  one_f1_score = sum(one_f1) / len(one_f1)
  one_support_score = sum(one_support) / len(one_support)

  macro_precision_score = sum(macro_precision) / len(macro_precision)
  macro_recall_score = sum(macro_recall) / len(macro_recall)
  macro_f1_score = sum(macro_f1) / len(macro_f1)
  macro_support_score = sum(macro_support) / len(macro_support)

  weighted_precision_score = sum(weighted_precision) / len(weighted_precision)
  weighted_recall_score = sum(weighted_recall) / len(weighted_recall)
  weighted_f1_score = sum(weighted_f1) / len(weighted_f1)
  weighted_support_score = sum(weighted_support) / len(weighted_support)

  zero_dict = {'precision': zero_precision_score, 'recall': zero_recall_score, 'f1-score': zero_f1_score, 'support': zero_support_score}
  one_dict = {'precision': one_precision_score, 'recall': one_recall_score, 'f1-score': one_f1_score, 'support': one_support_score}
  macro_dict = {'precision': macro_precision_score, 'recall': macro_recall_score, 'f1-score': macro_f1_score, 'support': macro_support_score}
  weighted_dict = {'precision': weighted_precision_score, 'recall': weighted_recall_score, 'f1-score': weighted_f1_score, 'support': weighted_support_score}

  combined_dict = {'0': zero_dict, '1': one_dict, 'macro avg': macro_dict, 'weighted avg': weighted_dict}

  class_report_df = pd.DataFrame(combined_dict).transpose()
  output_name = 'classification_report_all.csv'
  class_report_df.to_csv(os.path.join(label_output_path, output_name))


  # compile all data
  print(f"COMPILING ALL DATA")
  file_names = os.listdir(label_output_path)
  name = "classification_report"

  main_df = pd.DataFrame()
  for name in file_names:
    if 'classification_report' in name:
      temp_df = pd.read_csv(os.path.join(label_output_path, name))
      short_name = '_' + os.path.splitext(name)[0].split('classification_report_')[1]
      temp_df["Unnamed: 0"] = temp_df["Unnamed: 0"].astype(str) + short_name
      main_df = main_df.append(temp_df)


  main_df.to_csv(os.path.join(label_output_path, 'combined_classification_results.csv'))
  label_end = time.time()
  print(f"DONE PROCESSING LABEL: {label_name} IN {label_end - label_start}")


with open(raw_output_path, 'w', encoding='utf-8') as f:
  json.dump(predictions_main_dict, f, ensure_ascii=False, indent=4)

end_time = time.time()

print(f"TOTAL ELAPSED TIME: {end_time - start_time}")