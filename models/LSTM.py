from locale import nl_langinfo
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_size,self.hidden_size = configs.enc_in,configs.d_model
        self.num_layers = 1
        self.pred_len,self.feature_size = configs.pred_len,configs.c_out
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,self.num_layers,batch_first=True)
        self.fc = nn.Linear(configs.d_model, self.pred_len*self.feature_size)

    def forward(self, x):# [batch_size, seq_length, features]
        batch_size, seq_len = x.shape[0], x.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        output = self.fc(lstm_out[:, -1, :]) 
        # Use only the output from the last time step
        output = output.reshape(-1, self.pred_len, self.feature_size) 
        # [batch_size, pred_len, out]
        return output 