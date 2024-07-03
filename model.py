import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layer_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        # Adding additional LSTM layers
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(hidden_dim, hidden_dim, layer_dim, batch_first=True)
                for _ in range(layer_dim - 1)
            ]
        )

        # Initialize LSTM layers with Xavier initialization
        for lstm_layer in self.lstm_layers:
            nn.init.xavier_uniform_(lstm_layer.weight_ih_l0)
            nn.init.orthogonal_(lstm_layer.weight_hh_l0)

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Define softmax activation

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        for lstm_layer in self.lstm_layers:
            out, (hn, cn) = lstm_layer(out)
        return self.fc(self.softmax(out))
