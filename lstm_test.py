import numpy as np

import datasets

import torch 
import torch.nn as nn

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data_prep import TextDataset, vocab, train_labels, train_counts, padding_len

# Dictionary to convert from integer label to string
int_to_label = {
                0:        "sad",
                1:        "joy",
                2:        "love",
                3:        "anger",
                4:        "fear",
                5:        "suprise"
            }

# Tokenization function (same as data_prep.py)
def tokenize(sentence):
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

# Define LSTM model structure
class LSTM(nn.Module):

    # Initialization function with definitions for all layers
    def __init__(self, embedding_dim, vocab_len, num_labels, num_layers, dropout_prob):

        # Initialize parent class
        super().__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_len, 
                                      embedding_dim=embedding_dim,)
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=embedding_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout_prob,)
        # Linear Layer
        self.linout = nn.Linear(in_features=embedding_dim, 
                                out_features=num_labels)

        return
    
    # Forward pass of the networks, return output from last dense layer 
    def forward(self, x):

        # Obtain batch size dinamically
        batch_size = x.size(0)

        # Initialize hidden and cell states
        h0 = torch.zeros(num_layers, batch_size, embedding_dim)
        c0 = torch.zeros(num_layers, batch_size, embedding_dim)

        # Run input tensor through embedding layer
        x = self.embedding(x)

        # Run embedded tensor through LSTM layer (we ignore hidden and cell states as we won't be needing them)
        _, (final_hidden, _) = self.lstm(x, (h0, c0))

        return self.linout(final_hidden[-1])

# Directory for graphs
graph_dir = "./graphs/"

# Load test dataset
ds = datasets.load_dataset("dair-ai/emotion", "split")
ds_test = ds["test"]
# Obtain train split
ds_train = ds["train"]

# Dataloader hyperparameters
batch_size = 16

# Define test dataset
test_dataset = TextDataset(ds_test)
test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size, 
                          shuffle=False,
                          drop_last=False,)

# Model hyperparameters
embedding_dim = 64
num_layers = 2
dropout_prob = 0.5

# Initialize LSTM model
model = LSTM(   embedding_dim=embedding_dim,
                vocab_len=len(vocab), 
                num_labels=len(train_labels), 
                dropout_prob=dropout_prob, 
                num_layers=num_layers,          )

# Load pre-trained weights
model.load_state_dict(torch.load("./lstm_params.pt", weights_only=True))

# Set model to evaluation mode
model.eval()

# Compute class weights (to combad inbalanced dataset)
weights = torch.Tensor( len(ds_train) / np.array(train_counts) )

# Define loss criterion
criterion = nn.CrossEntropyLoss(weight=weights)

# Cumulative test loss and accuracy
test_loss = 0.0
test_accuracy = 0.0

# Itrate through all the batches in validation dataloader
for batch, (text, label) in enumerate(test_loader):

    # Don't compute gradients
    with torch.no_grad():
        # Compute output from model
        output = model(text)
        # Compute loss
        loss = criterion(output, label)
        # Add current batch loss to train loss
        test_loss += loss.item()
        # Add current batch accuracy to train accuracy
        test_accuracy += (output.argmax(1) == label).sum().item()/len(output)
        
# Compute average validation loss and accuracy
valid_loss = test_loss / len(test_loader)
valid_accuracy = 100 * test_accuracy / len(test_loader)

# Print accuracy and loss obtained
print("\n### MODEL ACCURACY ###")
print("Test Loss: \t", str(valid_loss))
print("Test Accuracy: \t", str(valid_accuracy))

# Print random batches and their prediction
for k in range(100):
    print("\n### RANDOM TEXT SAMPLE ###")
    index = np.random.randint(0, len(test_dataset))
    # Get original text
    text, label = test_dataset.show_original(index)
    print("\"" + str(text) + "\"", "\nLabel: " + str(label))
    # Prediction
    text, label = test_dataset[index]
    with torch.no_grad():
        output = model(text.unsqueeze(0))
    print("Model output: " + int_to_label[int(output.argmax(1))] )

# Classify text using user input
print("\n### USER INPUT ###")
print("Write a sentence to classify: ")
text = input()
# Split text into words
text = text.split(" ")
# Convert each word into integer using vocab (from train set), with padding previously calculated
text = tokenize(text)
# Convert tokenized text into a tensor
text = torch.from_numpy(text)
# Run it through model
with torch.no_grad():
    output = model(text.unsqueeze(0))
# Print output
print("Model output: " + int_to_label[int(output.argmax(1))] )