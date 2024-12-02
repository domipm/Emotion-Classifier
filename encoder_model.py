import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import datasets
import alive_progress

from torch.utils.data import DataLoader

# Import all encessary classes and variables from data_prer
from data_prep import TextDataset, vocab, train_labels, train_counts

import torch.nn as nn

# Dataloader hyperparameters
batch_size = 32

# Model hyperparameters
embedding_dim = 64
num_layers = 3
num_heads = 4
dropout_prob = 0.25

# Training hyperparameters
epochs = 25
learning_rate = 0.001
weight_decay = 0

class ENCODER(nn.Module):
    def __init__(self, embedding_dim, vocab_len, num_layers, num_heads, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.encoder = nn.TransformerEncoderLayer(d_model = embedding_dim, 
                                                        nhead=num_heads, 
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder,
                                            num_layers=num_layers,)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(embedding_dim, 1)
        
    def forward(self, x):
        # Run input tensor through embedding layer
        x = self.embedding(x)
        # Run embedded tensor through Transformed-Encoder layer
        x = self.encoder(x)
        # Apply dropout regularization
        x = self.dropout(x)
        # ???
        x = x.max(dim=1)[0]
        # Last linear layer that outputs predictions
        x = self.linear(x)
        # Return final tensor
        return x  

# Load dataset
ds = datasets.load_dataset("dair-ai/emotion", "split")
# Obtain train split
ds_train = ds["train"]

# Compute class weights (to combad inbalanced dataset)
weights = torch.Tensor( len(ds_train) / np.array(train_counts) )

# Create TextDataset class instance
train_dataset = TextDataset(ds_train)
# Create train dataloader
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True,)

text, label = train_dataset[0]

random = torch.rand(text.shape)
print(random.shape)

# Initialize LSTM model class instance
model = ENCODER(embedding_dim=embedding_dim,
                vocab_len=len(vocab), 
                num_layers=num_layers,
                num_heads=num_heads,
                dropout_prob=dropout_prob,)

output = model(text)
print(output)

exit()

# Initialize optimizer
optimizer = optim.Adam(params=model.parameters(), 
                       lr=learning_rate,
                       weight_decay=weight_decay)
# Loss function
criterion = nn.CrossEntropyLoss(weight=weights)

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
            print("pred", output[:,-1,:].argmax(dim=1))
            print("label", label)

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
    avg_train_accuracy = 100 * avg_train_accuracy/ len(train_loader) 
    # Update train loss over time
    train_loss_epochs.append(avg_train_loss)
    train_accuracy_epochs.append(avg_train_accuracy)

    # Print average loss for current epoch
    print('- Avg. Train Loss: {:.6f}\t Avg. Train Accuracy {:.6f}'.format(avg_train_loss, avg_train_accuracy))
