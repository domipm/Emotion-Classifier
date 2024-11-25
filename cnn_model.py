import torch
import torch.nn as nn

import datasets

from torch.utils.data import DataLoader

# Import all encessary classes and variables from data_prer
from data_prep import TextDataset, vocab, train_labels

# Define CNN model architecture
import torchsummary
import torch.nn as nn

class CNN(nn.Module):

    # Initialization function with definitions for all layers
    def __init__(self, embdedding_dim, vocab_len, num_labels, in_shape = None):

        # Initialize parent class
        super().__init__()
        # Define all layer to be used
        self.network = nn.Sequential(
                    # Embedding Layer
                    nn.Embedding(num_embeddings=vocab_len, embedding_dim=embdedding_dim),
                    # Output Linear Layer
                    nn.LazyLinear(out_features=len(num_labels)), 
        )

        # Print model summary if given input shape
        if in_shape != None:
            torchsummary.summary(self, in_shape)

        return
    
    # Forward pass of the networks, return output from last dense layer 
    def forward(self, x):

        return self.network(x)

# Load dataset
ds = datasets.load_dataset("dair-ai/emotion", "split")
# Obtain train split
ds_train = ds["train"]

# Create TextDataset class instance
train_dataset = TextDataset(ds_train)

# Example text and label dimensions
text, label = train_dataset[0]
print(text.shape)

exit()

# Create train dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize CNN model class instance
model = CNN(embdedding_dim=5, vocab_len=len(vocab), num_labels=len(train_labels), in_shape=text.shape)
