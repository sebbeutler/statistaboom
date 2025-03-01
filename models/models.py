import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
from models.lstm import LSTMModel

# 1. Download Yahoo Finance Data
def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock price data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if data is None:
        raise ValueError(f'No data found for {ticker} from {start_date} to {end_date}')
    return data['Close'].values  # Use the 'Close' price for the time series

# 2. Prepare Data for LSTM
def create_inout_sequences(input_data, seq_length):
    """
    Convert time series data into input-output sequences for LSTM.
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L - seq_length):
        train_seq = input_data[i:i + seq_length]
        train_label = input_data[i + seq_length:i + seq_length + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 4. Train the Model
def train_model(model, inout_seq, epochs=150, learning_rate=0.001):
    """
    Train the LSTM model.
    """
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for seq, labels in inout_seq:
            optimizer.zero_grad()

            # Convert NumPy arrays to PyTorch tensors
            seq_tensor = torch.FloatTensor(seq).unsqueeze(-1)  # Shape: (seq_length, 1)
            labels_tensor = torch.FloatTensor(labels)  # Shape: (1,)

            # Forward pass
            y_pred = model(seq_tensor.unsqueeze(0))  # Add batch dimension: (1, seq_length, 1)

            # Compute loss
            single_loss = loss_function(y_pred, labels_tensor)
            single_loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch} Loss: {single_loss.item()}')

# 5. Make Predictions
def make_predictions(model, data, seq_length):
    """
    Generate predictions for the entire dataset.
    """
    predictions = []
    for seq, _ in create_inout_sequences(data, seq_length):
        with torch.no_grad():
            # Convert NumPy array to PyTorch tensor
            seq_tensor = torch.FloatTensor(seq).unsqueeze(-1)  # Shape: (seq_length, 1)
            predictions.append(model(seq_tensor.unsqueeze(0)).item())  # Add batch dimension
    return predictions

# 6. Visualize Results
def plot_results(data, predictions, seq_length):
    """
    Plot the true data and predictions.
    """
    plt.plot(data, label='True Data')
    plt.plot(range(seq_length, len(data)), predictions, label='Predictions')
    plt.legend()
    plt.show()

def example():
    # Parameters
    ticker = "AAPL"  # Stock ticker (e.g., Apple Inc.)
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    seq_length = 20  # Length of input sequence
    epochs = 150
    learning_rate = 0.001

    # Step 1: Download stock data
    stock_data = download_stock_data(ticker, start_date, end_date)

    # Step 2: Prepare data for LSTM
    inout_seq = create_inout_sequences(stock_data, seq_length)
    print(len(inout_seq))
    return 0

    # Step 3: Initialize the model
    model = LSTMModel() # type: ignore

    # Step 4: Train the model
    train_model(model, inout_seq, epochs, learning_rate)

    # Step 5: Make predictions
    predictions = make_predictions(model, stock_data, seq_length)

    # Step 6: Visualize results
    plot_results(stock_data, predictions, seq_length)
