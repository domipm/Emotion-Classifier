import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import datasets
import alive_progress

from torch.utils.data import DataLoader

# Import all encessary classes and variables from data_prer
from data_prep import TextDataset, vocab, train_labels, train_counts

# Directory to save graphs
graph_dir = "./graphs/"

# Dataloader hyperparameters
batch_size = 16

# Model hyperparameters
embedding_dim = 64
num_layers = 2
dropout_prob = 0.5

# Training hyperparameters
epochs = 20
learning_rate = 0.00075
weight_decay = 0.0005

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

        # Initialize hidden and cell states
        h0 = torch.zeros(num_layers, batch_size, embedding_dim)
        c0 = torch.zeros(num_layers, batch_size, embedding_dim)

        # Run input tensor through embedding layer
        x = self.embedding(x)

        # Run embedded tensor through LSTM layer (we ignore hidden and cell states as we won't be needing them)
        _, (final_hidden, _) = self.lstm(x, (h0, c0))

        return self.linout(final_hidden[-1])

# Load dataset
ds = datasets.load_dataset("dair-ai/emotion", "split")
# Obtain train split
ds_train = ds["train"]
# Obtain validation split
ds_valid = ds["validation"]

# Compute class weights (to combad inbalanced dataset)
weights = torch.Tensor( len(ds_train) / np.array(train_counts) )

# Create TextDataset class instance for training
train_dataset = TextDataset(ds_train)
# Create train dataloader
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True,
                          drop_last=True,)

# Create TextDataset class instance for testing
valid_dataset = TextDataset(ds_valid)
# Create train dataloader
valid_loader = DataLoader(valid_dataset, 
                          batch_size=batch_size, 
                          shuffle=False,
                          drop_last=True,)

# Initialize LSTM model class instance
model = LSTM(   embedding_dim=embedding_dim,
                vocab_len=len(vocab), 
                num_labels=len(train_labels), 
                dropout_prob=dropout_prob, 
                num_layers=num_layers,          )

# Initialize optimizer
optimizer = optim.Adam(params=model.parameters(), 
                       lr=learning_rate,
                       weight_decay=weight_decay)
# Loss function
criterion = nn.CrossEntropyLoss(weight=weights)

# Train loss across epochs
train_loss_epochs = []
train_accuracy_epochs = []
# Test loss across epochs
valid_loss_epochs = []
valid_accuracy_epochs = []

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

            # Calculate loss
            loss = criterion(output, label)

            # Update train loss (cumulative) and accuracy
            avg_train_loss += loss.item()
            avg_train_accuracy += ( output.argmax(dim=1) == label ).sum().item()/len(output)

            # Set gradients to zero
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()
            # Perform optimizer step
            optimizer.step()

            # Update progress bar
            bar()

    # Update average train loss and accuracy
    avg_train_loss /= len(train_loader)
    avg_train_accuracy = 100 * avg_train_accuracy / len(train_loader) 
    # Update train loss over time
    train_loss_epochs.append(avg_train_loss)
    train_accuracy_epochs.append(avg_train_accuracy)

    # Print average loss for current epoch
    print('- Avg. Train Loss: {:.6f}\t Avg. Train Accuracy {:.6f}'.format(avg_train_loss, avg_train_accuracy))

    # Validation loss and accuracy
    avg_valid_loss = 0.0
    avg_valid_accuracy = 0.0

    # Set model to evaluation mode
    model.eval()

    # Ensure no gradient is computed
    with torch.no_grad():

        for batch, (text, label) in enumerate(valid_loader):
            
            # Output from model
            output = model(text)

            # Compute loss
            loss = criterion(output, label)

            # Add current batch loss to test loss
            avg_valid_loss += loss.item()
            # Add current batch accuracy to test accuracy
            avg_valid_accuracy += (output.argmax(dim=1) == label).sum().item()/len(output)

    # Compute average test loss for current epoch
    avg_valid_loss /= len(valid_loader)
    valid_loss_epochs.append(avg_valid_loss) 
    # Compute averate training accuracy for current epoch
    avg_valid_accuracy = 100 * avg_valid_accuracy / len(valid_loader)
    valid_accuracy_epochs.append(avg_valid_accuracy)

    # Print out average testing loss for each epoch 
    print('- Avg. Valid Loss: {:.6f}\t Avg. Valid Accuracy {:.6f}'.format(avg_valid_loss, avg_valid_accuracy), end="\n\n") 

# After training the model, save the parameters
torch.save(model.state_dict(),  "./lstm_params.pt")

# Write the training and testing loss and accuracy to file !
with open(graph_dir + "lstm_logs.txt", "w") as f:
    for epoch in range(epochs):
        f.write("{}\t{}\t{}\t{}\n".format(train_loss_epochs[epoch], valid_loss_epochs[epoch], train_accuracy_epochs[epoch], valid_accuracy_epochs[epoch]))
f.close()

# Plot training and testing loss over time
fig, ax = plt.subplots()

ax.plot(np.arange(epochs), train_loss_epochs, label="Train loss")
ax.plot(np.arange(epochs), valid_loss_epochs, label="Validation loss")

ax.set_title("LSTM Train/Validation Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Average epoch loss")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()
plt.savefig(graph_dir + "lstm_loss.png", dpi=300)
plt.show()
plt.close()

# Plot training and testing accuracy over time
fig, ax = plt.subplots()

ax.plot(np.arange(epochs), train_accuracy_epochs, label="Train accuracy")
ax.plot(np.arange(epochs), valid_accuracy_epochs, label="Validation accuracy")

ax.set_title("LSTM Train/Validation Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Average epoch accuracy [%]")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()
plt.savefig(graph_dir + "lstm_accuracy.png", dpi=300)
plt.show()
plt.close()