import torch
import torch.nn as nn
import torch.nn.functional as F

class RL2(nn.Module):
    """ Meta-learning model for the Categorisation task """

    def __init__(self, num_input, num_output, num_hidden, num_layers=1) -> None:
        super(RL2, self).__init__()
        self.num_input = num_input + num_output
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_input + num_output, num_hidden, num_layers, batch_first=True)
        self.linear = nn.Linear(num_hidden, num_output)  
        #self.init_weights()

    def forward(self, x, h, c):
        y, (h, c) = self.lstm(x, (h, c))
        y = self.linear(y)
        y = F.softmax(y, dim=-1)
        return y, h, c
    
    def make_inputs(self, inputs, prev_choices):
        return torch.cat([inputs.unsqueeze(1), F.one_hot(prev_choices, num_classes=self.num_output).unsqueeze(1)], dim=-1)
    
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