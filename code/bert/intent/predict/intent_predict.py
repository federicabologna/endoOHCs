print("starting process")
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
#model_name = 'distilbert-base-uncased'  

# Choose the GPU we want to process this script
device_name = 'cuda'       

# This is the maximum number of tokens in any document sent to BERT
max_length = 512      

endo_dir = '/share/luxlab/roz/endo'
output_path = os.path.join(endo_dir, 'output', 'predictions')
# dataset_path = os.path.join(endo_dir, 'endo+endometriosis.pkl') # data for predictions
dataset_path = os.path.join(endo_dir, 'endo+endometriosis.csv') # data for predictions

print("Reading in pickle file")
# endo_df = pd.read_pickle(dataset_path)
endo_df = pd.read_csv(dataset_path)
print("Read in file")

endo_df = endo_df[endo_df["type"] == "post"]
print("1")
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

  # set label-specific paths
  model_path = os.path.join(endo_dir, 'output', dir_name, 'model')
  raw_output_path = os.path.join(output_path, dir_name, 'raw_output')
  combined_output_path = os.path.join(output_path,  dir_name, 'combined_output')
  combined_csv_path = os.path.join(output_path, dir_name, 'combined_output.csv')

  batchsize = 32
  predictions = []
  print("2")
  # load the fine-tuned model from our directory and send it to cuda
  model = DistilBertForSequenceClassification.from_pretrained(model_path)#.to(device)
  print("3")
  # load the tokenizer (make sure this is the same type of tokenizer as what we used when training)
  tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
  print("4")
  # change input format
  # test_df = endo_df
  # test_df["shortened_post"] = test_df["text"].apply(get_sentence_segments, sections = input_part, n = input_length)
  print("5")
  worklist = endo_df["text"].to_list()

  for i in range(0, len(worklist), batchsize):
      batch = worklist[i:i+batchsize] # extract batch from worklist
      test_encodings = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")#.to(device) # tokenize the posts
      output = model(**test_encodings) # make predictions with model on our test_encodings for this batch
      batch_predictions = torch.softmax(output.logits, dim=1).tolist() # get the predictions result
      predictions.append(batch_predictions)
      if i % 1000 == 0:
          print(str(i)+" in "+str(len(worklist)))

  pickle.dump(predictions, open(raw_output_path, "wb"))

  col_name = "predictions_"+dir_name

  # add predictions to main df
  flat_list = [item for sublist in predictions for item in sublist]
  endo_df["predictions"] = flat_list
  endo_df[['prob_0','prob_1']] = pd.DataFrame(endo_df["predictions"].tolist(), index=endo_df.index)
  endo_df[col_name] = np.where(endo_df['prob_1'] > .50, 1, 0) # this is the column we're interested in, since this is a binary label

  # TO DO: figure out combined output path for pickling
  endo_df.to_pickle(combined_output_path)

  # TO DO: figure out csv path
  endo_df.to_csv(combined_csv_path)
  label_end = time.time()

  print(f"LABEL {label_name} FINISHED PROCESSING IN {label_end - label_start} SECONDS.")

# total end time
end_time = time.time()
print(f"ALL PREDICTIONS FINISHED PROCESSING IN {end_time - start_time} SECONDS.")

