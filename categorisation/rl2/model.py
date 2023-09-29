import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_utils import PositionalEncoding

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
        return torch.zeros(self.num_layers, batch_size, self.num_hidden), \
            torch.zeros(self.num_layers, batch_size, self.num_hidden)

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
        return torch.zeros(self.num_layers, batch_size, self.num_hidden), \
            torch.zeros(self.num_layers, batch_size, self.num_hidden)

    def compute_loss(self, model_choices, true_choices):
        return self.criterion(model_choices, true_choices)
    
class Transformer(nn.Module):
    """ Meta-learning model with transformer core for the categorisation task """

    def __init__(self, num_input, num_output, num_hidden, num_layers=1, d_model=256, num_head=1, dropout=0.1, beta=1, max_steps=200) -> None:
        super(Transformer, self).__init__()
        
        self.num_input = num_input + num_output
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_layers = num_layers

         
        # check d_model is correct
        assert d_model % num_head == 0, "nheads must divide evenly into d_model"
        
        # embedding layer to move to d_model space
        self.embedding = nn.Linear(self.num_input, d_model)

        # position encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=max_steps,
        )

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head, \
                                                   dim_feedforward=num_hidden, \
                                                    dropout=dropout, batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # linear layer 
        self.linear = nn.Linear(d_model, num_output)  
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.beta = beta

    def forward(self, packed_inputs, sequence_lengths, mask=None):
            
            #packed_inputs = pack_padded_sequence(packed_inputs, sequence_lengths, batch_first=True, enforce_sorted=False)
            inputs = self.embedding(packed_inputs)
            inputs_pos_encoded = self.pos_encoder(inputs)
            output = self.transformer(inputs_pos_encoded.float(), mask=mask)
            #output, _ = pad_packed_sequence(packed_output, batch_first=True)
            y = self.linear(output)
            y = self.sigmoid(self.beta*y)
            
            return y
    
    def make_inputs(self, inputs, prev_choices):
        raise NotImplementedError
        #TODO: concatenate inputs and previous_choices of all trials until then

    def initial_states(self, batch_size):
        return None, None
    
    def compute_loss(self, model_choices, true_choices):
        return self.criterion(model_choices, true_choices)
    

class TransformerDecoder(nn.Module):
    """ Meta-learning model with transformer core for the categorisation task """

    def __init__(self, num_input, num_output, num_hidden, num_layers=1, d_model=256, num_head=1, dropout=0.1, beta=1, max_steps=200) -> None:
        super(TransformerDecoder, self).__init__()
        
        self.num_input = num_input + num_output
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_layers = num_layers

         
        # check d_model is correct
        assert d_model % num_head == 0, "nheads must divide evenly into d_model"
        
        # embedding layer to move to d_model space
        self.embedding = nn.Linear(self.num_input, d_model)

        # position encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=max_steps,
        )

        # transformer encoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_head, \
                                                   dim_feedforward=num_hidden, \
                                                    dropout=dropout, batch_first=True)
        
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # linear layer 
        self.linear = nn.Linear(d_model, num_output)  
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.beta = beta

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, packed_inputs, sequence_lengths, mask=None):
            
            #packed_inputs = pack_padded_sequence(packed_inputs, sequence_lengths, batch_first=True, enforce_sorted=False)
            inputs = self.embedding(packed_inputs)
            inputs_pos_encoded = self.pos_encoder(inputs)
            tgt_mask = self.generate_square_subsequent_mask(inputs_pos_encoded.size(1))
            output = self.transformer(inputs_pos_encoded.float(), inputs_pos_encoded.float(), tgt_mask=tgt_mask, memory_mask=tgt_mask)
            #output, _ = pad_packed_sequence(packed_output, batch_first=True)
            y = self.linear(output)
            y = self.sigmoid(self.beta*y)
            
            return y
    
    def make_inputs(self, inputs, prev_choices):
        raise NotImplementedError
        #TODO: concatenate inputs and previous_choices of all trials until then

    def initial_states(self, batch_size):
        return None, None

    
    def compute_loss(self, model_choices, true_choices):
        return self.criterion(model_choices, true_choices)
