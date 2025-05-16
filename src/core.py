from _input import extract_features
from train import *
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.rename('test_sequences.csv', 'pred-seqs.csv')

pre-train_sequences_df = extract_features('train_sequences.csv')
split_index = int(0.96 * len(pre-train_sequences_df))
train_sequences_df = pre-train_sequences_df[:split_index]
test_sequences_df = pre-train_sequences_df[split_index:]

train_sequences_df.to_csv('train_sequences.csv', index=False)
test_sequences_df.to_csv('test_sequences.csv', index=False)

pre-train_labels_df = pd.read_csv('train_labels.csv')
train_labels_df = pre-train_labels_df[:split_index]
test_labels_df = pre-train_labels_df[split_index:]

train_labels_df.to_csv('train_labels.csv', index=False)
test_labels_df.to_csv('test_labels.csv', index=False)


validation_sequences_df = extract_features('validation_sequences.csv')
validation_sequences_df.to_csv('validation_sequences.csv', index=False)





