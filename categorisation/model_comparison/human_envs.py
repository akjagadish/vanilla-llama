import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils


class Badham2017(nn.Module):
    """
    load human data from Badham et al. 2017
    """
    
    def __init__(self, noise=0., return_prototype=False, device='cpu'):
        super(Badham2017, self).__init__()
        
        self.device = torch.device(device)
        self.data = pd.read_csv('/u/ajagadish/vanilla-llama/categorisation/data/human/badham2017deficits.csv')
        self.num_choices = 1 
        self.num_dims = 3
        self.noise = noise
        self.return_prototype = return_prototype

    def sample_batch(self, participant):
    
        stacked_task_features, stacked_targets, stacked_human_targets, stacked_prototypes = self.get_participant_data(participant)
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(stacked_task_features, batch_first=True)

        if self.return_prototype:
            return packed_inputs, sequence_lengths, stacked_targets, stacked_human_targets, stacked_prototypes
        else:
            return packed_inputs, sequence_lengths, stacked_targets, stacked_human_targets

    def get_participant_data(self, participant):
        
        inputs_list, targets_list, human_targets_list, prototype_list = [], [], [], []
    
        # get data for the participant
        data_participant = self.data[self.data['participant']==participant]
        conditions = np.unique(data_participant['condition'])
        for task_type in conditions:
            
            # get features and targets for the task
            input_features = np.stack([eval(val) for val in data_participant[data_participant.condition==task_type].all_features.values])
            human_choices = data_participant[data_participant.condition==task_type].choice
            true_choices = data_participant[data_participant.condition==task_type].correct_choice
            
            # convert human choices to 0s and 1s
            targets = np.array([1. if choice=='j' else 0. for choice in true_choices])
            human_targets = np.array([1. if choice=='j' else 0. for choice in human_choices])

            # flip features, targets and humans choices 
            if np.random.rand(1) > 0.5:
                input_features = 1. - input_features
                targets = 1. - targets  
                human_targets = 1. - human_targets

            # concatenate all features and targets into one array with placed holder for shifted target
            sampled_data = np.concatenate((input_features, targets.reshape(-1, 1), targets.reshape(-1, 1)), axis=1)
            
            # replace placeholder with shifted targets to the sampled data array
            sampled_data[:, self.num_dims] = np.concatenate((np.array([0. if np.random.rand(1) > 0.5 else 1.]), sampled_data[:-1, self.num_dims]))
            
            # stacking all the sampled data across all tasks
            inputs_list.append(torch.from_numpy(sampled_data[:, :(self.num_dims+1)]))
            targets_list.append(torch.from_numpy(sampled_data[:, [self.num_dims+1]]))
            human_targets_list.append(torch.from_numpy(human_targets.reshape(-1, 1)))

            # compute mean of each features for a category
            prototype_list.append([np.mean(input_features[targets==0], axis=0), np.mean(input_features[targets==1], axis=0)])

     
        return inputs_list, targets_list, human_targets_list, prototype_list  
       