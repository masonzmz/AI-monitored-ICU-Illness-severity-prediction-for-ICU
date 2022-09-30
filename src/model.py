import torch


class GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, hidden_layers,
                 output_size, batch_size, dropout, device):
        super().__init__() 
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layers = hidden_layers

        self.rnn = torch.nn.GRU(input_size, hidden_layer_size, hidden_layers, dropout=dropout, batch_first=True)
        self.l1 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l2 = torch.nn.Linear(hidden_layer_size, output_size)

        self.relu = torch.nn.ReLU()
        self.hidden_cell = torch.zeros(hidden_layers, batch_size, self.hidden_layer_size).to(device)

    def forward(self, input_seq):
        gru_out, self.hidden_cell = self.rnn(input_seq, self.hidden_cell)
        l1_out = self.l1(gru_out) 
        l1_out = self.relu(l1_out)
        predictions = self.l2(l1_out)
        
        return predictions[:, -1, :]

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, hidden_layers,
                 output_size, batch_size, dropout, device):
        super().__init__() 
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layers = hidden_layers

        self.rnn = torch.nn.RNN(input_size, hidden_layer_size, hidden_layers, dropout=dropout, batch_first=True)
        self.l1 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l2 = torch.nn.Linear(hidden_layer_size, output_size)

        self.relu = torch.nn.ReLU()
        self.hidden_cell = torch.zeros(hidden_layers, batch_size, self.hidden_layer_size).to(device)

    def forward(self, input_seq):
        rnn_out, self.hidden_cell = self.rnn(input_seq, self.hidden_cell)
        l1_out = self.l1(rnn_out) 
        l1_out = self.relu(l1_out)
        predictions = self.l2(l1_out)
        
        return predictions[:, -1, :]

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, hidden_layers,
                 output_size, batch_size, dropout, device):
        super().__init__() 
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layers = hidden_layers

        self.rnn = torch.nn.LSTM(input_size, hidden_layer_size, hidden_layers, dropout=dropout, batch_first=True)
        self.l1 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l2 = torch.nn.Linear(hidden_layer_size, output_size)

        self.relu = torch.nn.ReLU()
        self.hidden_cell = (torch.zeros(hidden_layers, batch_size, self.hidden_layer_size).to(device),
                            torch.zeros(hidden_layers, batch_size, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.rnn(input_seq, self.hidden_cell)
        l1_out = self.l1(lstm_out) 
        l1_out = self.relu(l1_out)
        predictions = self.fc2(l1_out)
        
        return predictions[:, -1, :]
