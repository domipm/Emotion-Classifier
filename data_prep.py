# Script to perform all necessary data preparation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from datasets import load_dataset

# Graph directory to save all figures
graph_dir = "./visualizations/"
# Tensor directory
tensor_dir = "./tensors/"

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
        plt.savefig(graph_dir + split + "_distr.png", dpi=300, bbox_inches="tight")
        plt.close()

    return labels, counts

# Gather the train, test, and validation label class distribution
train_labels, train_counts = get_labels(dataset=ds_train, split="train")
test_labels, test_counts = get_labels(dataset=ds_test, split="test")
valid_labels, valid_counts = get_labels(dataset=ds_valid, split="validation")

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
plt.savefig(graph_dir + "data_stats.png", dpi=300, bbox_inches="tight")
plt.close()

# Get all words in train dataset as one-dimensional list
all_words = [word for sentence in train_split for word in sentence]
# Construct dictionary out of this list: vocabulary of dataset
vocab = {word: index for index, word in enumerate(all_words, start=1)}
# Add padding token (PAD) and not-a-word token (NAW)
vocab.update({"PAD": 0, "NAW": -1})

# Define padding length (as maximum sentence lenght of all datasets)
padding_len = np.max((train_stats[3], test_stats[3], valid_stats[3]))

# dataset_split refers to the already-split (by " " blank space) dataset
def tokenizer(dataset_split, vocab=vocab, padding_len=padding_len):
    "Function that tokenizes complete dataset into sequences of tokens (integers)"

    dataset_tokenized = []
    # For each sentence, indexed by k with list of string sentence
    for k, sentence in enumerate(dataset_split):
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
        # Append sentence to dataset
        dataset_tokenized.append(sentence_tokenized)
    # Return tokenized dataset
    return dataset_tokenized

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

# Save tensors into file (so we can load them in model script)
torch.save(train_tensor, tensor_dir + "train.pt")
torch.save(test_tensor, tensor_dir + "test.pt")
torch.save(valid_tokenized, tensor_dir + "valid.pt")