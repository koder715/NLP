import torch
import torch.nn as nn


class CharRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        embedded_copy = embedded.clone().detach()
        hidden_copy = (hidden[0].clone().detach(), hidden[1].clone().detach())  # Make a detached copy of the hidden state
        lstm_out, hidden = self.lstm(embedded_copy, hidden_copy)  # Use the detached copy of hidden state
        output = self.fc(lstm_out)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


# Create an instance of the model
model = CharRNN(input_size=100, hidden_size=128, output_size=100, num_layers=2)

# Generate some random input
x = torch.randint(0, 100, (64, 10))  # Random input of shape (batch_size=64, seq_length=10)
hidden = model.init_hidden(64)  # Initial hidden state

# Perform a forward pass
output, hidden = model(x, hidden)

# Check for inplace operations
for name, module in model.named_children():
    if hasattr(module, 'forward'):
        print(f"Checking module: {name}")
        for op_name, op in module.named_children():
            print(f"  Checking operation: {op_name}")
            with torch.no_grad():
                # Make a copy of the input
                x_copy = x.clone().detach()
                # Perform a forward pass with a copy of the input
                _ = op(x_copy)
            # Check if the original input has been modified
            inplace = torch.equal(x, x_copy)
            print(f"    Inplace: {inplace}")
