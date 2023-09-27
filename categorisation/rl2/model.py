import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MetaLearner(nn.Module):
    """ Meta-learning model with LSTM core for the categorisation task """

    def __init__(self, num_input, num_output, num_hidden, num_layers=1) -> None:
        super(MetaLearner, self).__init__()
        self.num_input = num_input + num_output
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_input + num_output, num_hidden, num_layers, batch_first=True)
        self.linear = nn.Linear(num_hidden, num_output)  
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        #self.init_weights()

    def forward(self, packed_inputs, sequence_lengths):

        #lengths, sort_idx = torch.sort(torch.tensor(sequence_lengths), descending=True) #.sort(reverse=True)
        #packed_inputs = packed_inputs[sort_idx]
        lengths = sequence_lengths
        packed_inputs = pack_padded_sequence(packed_inputs, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_inputs.float())
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # _, unsort_idx = sort_idx.sort()
        #output = output[unsort_idx]
        y = self.linear(output)
        y = self.sigmoid(y)
        
        return y
    
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
        return self.criterion(model_choices, true_choices)

class NoisyMetaLearner(nn.Module):
    """ Meta-learning model with LSTM core and an explicit tempature term for the categorisation task """

    def __init__(self, num_input, num_output, num_hidden, beta=1., num_layers=1) -> None:
        super(NoisyMetaLearner, self).__init__()
        self.num_input = num_input + num_output
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_input + num_output, num_hidden, num_layers, batch_first=True)
        self.linear = nn.Linear(num_hidden, num_output)  
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.beta = beta

    def forward(self, packed_inputs, sequence_lengths):


        lengths = sequence_lengths
        packed_inputs = pack_padded_sequence(packed_inputs, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_inputs.float())
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        y = self.linear(output)
        y = self.sigmoid(self.beta*y)
        
        return y
    
    def make_inputs(self, inputs, prev_choices):
        return torch.cat([inputs.unsqueeze(1), F.one_hot(prev_choices, num_classes=self.num_output).unsqueeze(1)], dim=-1)
    
    def initial_states(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.num_hidden), torch.zeros(self.num_layers, batch_size, self.num_hidden)

    def compute_loss(self, model_choices, true_choices):
        return self.criterion(model_choices, true_choices)
    
