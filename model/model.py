import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # Output layer 1
        self.fc2 = nn.Linear(64, 32) 
        self.fc3 = nn.Linear(32, 1)                # Output layer 2
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Nhân đôi num_layers vì mỗi chiều có 2 hidden layers
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size * 2)
        
        out = out[:, -1, :]
        
        # Output layers
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out.squeeze(1)