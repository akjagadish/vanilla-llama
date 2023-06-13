import torch
import torch.nn as nn
import torch.nn.functional as F

class RL2(nn.Module):
    """ Meta-learning model for the Categorisation task """

    def __init__(self, num_input, num_output, num_hidden, num_layers=1) -> None:
        super(RL2, self).__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_input, num_hidden, num_layers, batch_first=True)
        self.linear = nn.Linear(num_hidden, num_output)  
        #self.init_weights()

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        return x, h, c
    
    # def init_weights(self):
    #     for name, param in self.named_parameters():
    #         if 'bias' in name:
    #             nn.init.constant_(param, 0.0)
    #         elif 'weight' in name:
    #             nn.init.xavier_normal_(param)

    def initial_states(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.num_hidden), torch.zeros(self.num_layers, batch_size, self.num_hidden)

    def compute_loss(self, model_choices, true_choices):
        return F.cross_entropy(model_choices, true_choices)