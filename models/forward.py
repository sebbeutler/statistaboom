import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(RNNModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_layer_size, batch_first=True)

        # Fully connected layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Initialize hidden state
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size)

        # Pass through RNN
        rnn_out, _ = self.rnn(input_seq, h0)

        # Pass through fully connected layer
        predictions = self.linear(rnn_out[:, -1])
        return predictions
