import torch
import torch.nn as nn
import torch.optim as optim

import datasets
import alive_progress

from torch.utils.data import DataLoader

# Import all encessary classes and variables from data_prer
from data_prep import TextDataset, vocab, train_labels

import torchsummary
import torch.nn as nn

# Dataloader hyperparameters
batch_size = 32

# Model hyperparameters
embedding_dim = 64
num_layers = 5
dropout_prob = 0.5

# Training hyperparameters
epochs = 15
learning_rate = 0.0001
weight_decay = 0.00001

# Define LSTM model structure
class LSTM(nn.Module):

    # Initialization function with definitions for all layers
    def __init__(self, embedding_dim, vocab_len, num_labels, num_layers, dropout_prob, in_shape = None):

        # Initialize parent class
        super().__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_len, 
                                      embedding_dim=embedding_dim)
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=embedding_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout_prob)
        # Linear Layer
        self.linout = nn.Linear(in_features=embedding_dim, 
                                out_features=num_labels)

        # Print model summary if given input shape
        if in_shape != None:
            torchsummary.summary(self, in_shape)

        return
    
    # Forward pass of the networks, return output from last dense layer 
    def forward(self, x):
        # Run input tensor through embedding layer
        input_embeddings = self.embedding(x)
        # Run embedded tensor through LSTM layer (we ignore hidden and cell states as we won't be needing them)
        output, *_ = self.lstm(input_embeddings)
        # Last linear layer that outputs predictions for each class
        output = self.linout(output)
        # Return final tensor
        return output

# Load dataset
ds = datasets.load_dataset("dair-ai/emotion", "split")
# Obtain train split
ds_train = ds["train"]

# Create TextDataset class instance
train_dataset = TextDataset(ds_train)
# Create train dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize CNN model class instance
model = LSTM(   embedding_dim=embedding_dim,
                vocab_len=len(vocab), 
                num_labels=len(train_labels), 
                dropout_prob=dropout_prob, 
                num_layers=num_layers           )

# Initialize optimizer
optimizer = optim.Adam(params=model.parameters(), 
                       lr=learning_rate,
                       weight_decay=weight_decay)
# Loss function
criterion = nn.CrossEntropyLoss()

# Train loss across epochs
train_loss_epochs = []
train_accuracy_epochs = []

# Run over all training epochs
for epoch in range(epochs):

    # Set model to training mode
    model.train()

    # Training loss and accuracy
    avg_train_loss = 0.0
    avg_train_accuracy = 0.0

    # Bar for training progress visualization
    with alive_progress.alive_bar(total=len(train_loader),
                   title="Epoch {}/{}".format(epoch+1, epochs),
                   bar="classic", 
                   spinner=None, 
                   monitor="Batch {count}/{total}", 
                   elapsed="[{elapsed}]",
                   elapsed_end="[{elapsed}]",
                   stats=None) as bar:

        # Iterate through traininig dataset batches
        for batch, (text, label) in enumerate(train_loader):

            # Forward pass through the model
            output = model(text)

            # Calculate the loss (select last hidden state [:,-1,:] for final output)
            loss = criterion(output[:, -1, :], label)

            # Update train loss (cumulative) and accuracy
            avg_train_loss += loss
            avg_train_accuracy += ( output[:, -1, :].argmax(dim=1) == label ).sum().item()/len(output)
            print(output[:,-1,:].argmax(dim=1))
            print(label)
            # Set gradients to zero
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()
            # Perform optimizer step
            optimizer.step()

            bar()

    # Update average train loss and accuracy
    avg_train_loss /= len(train_loader)
    avg_train_accuracy = 100 * avg_train_accuracy/ len(train_loader) 
    # Update train loss over time
    train_loss_epochs.append(avg_train_loss)
    train_accuracy_epochs.append(avg_train_accuracy)

    # Print average loss for current epoch
    print('- Avg. Train Loss: {:.6f}\t Avg. Train Accuracy {:.6f}'.format(avg_train_loss, avg_train_accuracy))
