# Script to perform all necessary data preparation
# This also includes dataset class at the end

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from torch.utils.data import Dataset

from datasets import load_dataset

# Graph directory to save all figures
out_dir = "./data_prep/"

# Load the dataset to use
ds = load_dataset("dair-ai/emotion", "split")

# Obtain the train, test, and validation splits
ds_train = ds["train"]
ds_test = ds["test"]
ds_valid = ds["validation"]

# Define dictionaries to convert between string label and integer (depends on dataset!)

# Dictionary to convert from string label to integer
label_to_int = {
                "sad":        0,
                "joy":        1,
                "love":       2,
                "anger":      3,
                "fear":       4,
                "suprise":    5
            }
# Dictionary to convert from integer label to string
int_to_label = {
                0:        "sad",
                1:        "joy",
                2:        "love",
                3:        "anger",
                4:        "fear",
                5:        "suprise"
            }
# Dictionary to convert from integer label to color (visualization)
int_to_color = {
                0:        "tab:gray",
                1:        "tab:blue",
                2:        "tab:pink",
                3:        "tab:red",
                4:        "tab:orange",
                5:        "tab:olive"
            }

# Obtain label frequency in dataset, plot visualizations
def get_labels(dataset, split, plot = True):
    " Function that returns array of all labels and their count in dataset"

    # Obtain labels for all sentences
    all_labels = [ dataset[k]["label"] for k in range(len(dataset)) ]
    # Obtain unique and ordered labels
    labels = list(set(all_labels))
    # Obtain label count for the dataset
    counts = [all_labels.count(label) for label in labels]

    # Plot visualization if necessary
    if plot == True:
        bars = plt.bar(labels, counts/np.full(len(counts), len(dataset))*100, label="hi")
        for k, bar in enumerate(bars):
            plt.text(bar.get_x() + 0.05, bar.get_height() + 0.35, s=int_to_label[ labels[k] ])
            bar.set_color(int_to_color[ labels[k] ])
        plt.title(split + " split label class distribution")
        plt.xlabel("Label")
        plt.ylabel("Occurrence [%]")
        plt.savefig(out_dir + split + "_distr.png", dpi=300, bbox_inches="tight")
        plt.close()

    return labels, counts

# Gather the train, test, and validation label class distribution
train_labels, train_counts = get_labels(dataset=ds_train, split="train")
test_labels, test_counts = get_labels(dataset=ds_test, split="test")
valid_labels, valid_counts = get_labels(dataset=ds_valid, split="validation")

# Label lists
train_labellist = np.array( [ ds_train[k]["label"] for k in range(len(ds_train)) ] )
test_labellist = np.array( [ ds_test[k]["label"] for k in range(len(ds_test)) ] )
valid_labellist = np.array( [ ds_valid[k]["label"] for k in range(len(ds_valid)) ] )

# Convert these to tensors
train_labellist_tensor = torch.from_numpy(train_labellist)
test_labellist_tensor = torch.from_numpy(test_labellist)
valid_labellist_tensor = torch.from_numpy(valid_labellist)

# Obtain statistics of each dataset
def get_stats(dataset, splitting=" "):

    # List for each sentence in dataset of lists of words
    dataset_split = [ dataset[sentence]["text"].split(splitting) for sentence in range(len(dataset)) ]
    # Calculate statistics of dataset
    lengths = [ len(sentence) for sentence in dataset_split ]
    # Compute average, standard deviation, minimum and maximum for lengths
    dataset_stats = [ np.average(lengths), np.std(lengths), np.min(lengths), np.max(lengths) ]

    return dataset_split, dataset_stats

# Calculate split dataset and statistics for each split
train_split, train_stats = get_stats(ds_train)
test_split, test_stats = get_stats(ds_test)
valid_split, valid_stats = get_stats(ds_valid)

# Plot statistics for each dataset split
splits = ("Train", "Test", "Valid")
values = {
    'Avg. Len.': (train_stats[0], test_stats[0], valid_stats[0]),
    'Std. Len.': (train_stats[1], test_stats[1], valid_stats[1]),
    'Min. Len.': (train_stats[2], test_stats[2], valid_stats[2]),
    'Max. Len.': (train_stats[3], test_stats[3], valid_stats[3]),
}
x = np.arange(len(splits))
width = 0.2
multiplier = 0
fig, ax = plt.subplots(layout='constrained')
# Plot all values as bar plots
for attribute, measurement in values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, fmt="{:.02f}", fontsize=8)
    multiplier += 1
ax.set_xticks(x + width, splits)
# Save figure
plt.title("Dataset split statistics")
plt.legend()
plt.savefig(out_dir + "data_stats.png", dpi=300, bbox_inches="tight")
plt.close()

# Get all words in train dataset as one-dimensional list
all_words = [word for sentence in train_split for word in sentence]
# Get all unique, sorted word list in train dataset
all_words = sorted(list(set(all_words)))
# Construct vocabulary, padding (PAD) and not-a-word (NAW) tokens
vocab = {"PAD": 0, "NAW": 1}
# Construct dictionary out of train dataset
vocab.update( {word: index for index, word in enumerate(all_words, start=2)} )

# Save vocabulary as json file
import json
with open(out_dir + "vocab.json", "w") as f:
    f.write(json.dumps(vocab))

# Define padding length (as maximum sentence lenght of all datasets)
padding_len = np.max((train_stats[3], test_stats[3], valid_stats[3]))
# Define padding length (as average sentence length)
padding_len = int( np.average(( train_stats[0], test_stats[0], valid_stats[0] )) )

# dataset_split refers to the already-split (by " " blank space) dataset
def tokenizer(dataset_split, vocab=vocab, padding_len=padding_len):
    "Function that tokenizes complete dataset into sequences of tokens (integers)"
    dataset_tokenized = []
    # For each sentence, indexed by k with list of string sentence
    for sentence in dataset_split:
        sentence_tokenized = []
        # For each word in this sentence
        for word in sentence:
            # Convert word into integer token using vocab
            try: sentence_tokenized.append( vocab[word] )
            # If word not in sentence, use NAW padding
            except: sentence_tokenized.append( vocab["NAW"] )
        # Apply padding if necessary
        if len(sentence_tokenized) < padding_len:
            # Calculate how many padding tokens to add at the end of sentence
            pad = padding_len - len(sentence_tokenized)
            sentence_tokenized += [ vocab["PAD"] ] * pad
        # Truncate all sentences to padded length
        sentence_tokenized = sentence_tokenized[0:padding_len]
        # Append sentence to dataset
        dataset_tokenized.append(sentence_tokenized)
    # Return tokenized dataset
    return dataset_tokenized

# Convert each dataset split into final tensors (just to check it works)

# Tokenize all three datasets
train_tokenized = tokenizer(train_split)
test_tokenized = tokenizer(test_split)
valid_tokenized = tokenizer(valid_split)
# Convert these into numpy arrays
train_tokenized = np.array(train_tokenized)
test_tokenized = np.array(test_tokenized)
valid_tokenized = np.array(valid_tokenized)
# Convert each tokenized dataset into a tensor
train_tensor = torch.from_numpy(train_tokenized)
test_tensor = torch.from_numpy(test_tokenized)
vaid_tensor = torch.from_numpy(valid_tokenized)

# Dataset class
class TextDataset(Dataset):
    
    # Initialization function
    def __init__(self, dataset):
        # Initialize parent modules
        super().__init__()
        # Define dataset
        self.dataset = dataset
        return
    
    # Return length of dataset (number of sentences)
    def __len__(self):
        return len(self.dataset)
    
    # Return tokenized and padded text along with its label
    def __getitem__(self, index):
        # Load text and label from dataset
        text = self.dataset[index]["text"]
        label = self.dataset[index]["label"]
        # Convert label to tensor object
        label = torch.tensor(label, dtype=torch.long)
        # Split text into words
        text = text.split(" ")
        # Convert each word into integer using vocab (from train set), with padding previously calculated
        text = self.tokenize(text)
        # Convert tokenized text into a tensor
        text = torch.from_numpy(text)
        # Return tokenized text and label
        return text, label

    def tokenize(self, sentence):
        "Function to tokenize single sentence"
        sentence_tokenized = []
        # For each word in this sentence
        for word in sentence:
            # Convert word into integer token using vocab
            try: sentence_tokenized.append( vocab[word] )
            # If word not in sentence, use NAW padding
            except: sentence_tokenized.append( vocab["NAW"] )
        # Apply padding if necessary
        if len(sentence_tokenized) < padding_len:
            # Calculate how many padding tokens to add at the end of sentence
            pad = padding_len - len(sentence_tokenized)
            sentence_tokenized += [ vocab["PAD"] ] * pad
        # Truncate all sentences to padded length
        sentence_tokenized = sentence_tokenized[0:padding_len]
        # Return tokenized dataset
        return np.array(sentence_tokenized)
    
    def show_original(self, index):
        "Function to show original, non-tokenized sentence for given index"

        # Load text and label from dataset
        text = self.dataset[index]["text"]
        label = self.dataset[index]["label"]
        # Convert label index to string
        label = int_to_label[label]
        # Return text and label (strings)
        return text, label