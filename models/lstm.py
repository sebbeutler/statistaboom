import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        # Fully connected layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Initialize hidden state
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size)
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size)

        # Pass through LSTM
        lstm_out, _ = self.lstm(input_seq, (h0, c0))

        # Pass through fully connected layer
        predictions = self.linear(lstm_out[:, -1])
        return predictions
