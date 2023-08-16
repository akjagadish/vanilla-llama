import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils

class CategorisationTask(nn.Module):
    """
    Categorisation task inspired by Shepard et al. (1961)
    Note: generates one task at a time, each containing max_steps datapoints, with no repitition of datapoints over blocks
    """
    def __init__(self, data, max_steps=8, num_dims=3, batch_size=64, mode='train', split=[0.8, 0.1, 0.1], device='cpu', synthetic_data=False, num_tasks=10000, noise=0.1, shuffle_trials=False): 
        """ 
        Initialise the environment
        Args: 
            data: path to csv file containing data
            max_steps: number of steps in each episode
            num_dims: number of dimensions in each input
            batch_size: number of tasks in each batch
        """
        super(CategorisationTask, self).__init__()
        self.data = pd.read_csv(data)
        self.device = torch.device(device)
        self.num_choices = 1 #self.data.target.nunique()
        #TODO: max steps is equal to max_steps in the dataset
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.mode = mode
        self.split = (torch.tensor([split[0], split[0]+split[1], split[0]+split[1]+split[2]]) * self.data.task_id.nunique()).int()
        self.synthetic_data = synthetic_data
        self.noise = noise
        self.shuffle_trials = shuffle_trials
        if synthetic_data:
            self.generate_synthetic_data(num_tasks, split)

    def return_tasks(self, mode=None):
        mode = self.mode if mode is None else mode
        if mode == 'train':
            tasks = np.random.choice(self.data.task_id.unique()[:self.split[0]], self.batch_size, replace=False)
        elif mode == 'val':
            self.batch_size = self.split[1] - self.split[0]
            tasks = self.data.task_id.unique()[self.split[0]:self.split[1]]
        elif mode == 'test':
            self.batch_size = self.split[2] - self.split[1]
            tasks = self.data.task_id.unique()[self.split[1]:]
        
        return tasks
    def reset(self):
        """
        Reset the environment
        Returns:
            inputs: the inputs for the current step
            targets: the targets for the current step
            done: whether the episode is done
            info: additional information
        """
        if self.synthetic_data:
            inputs, targets = self.get_synthetic_data()
        else: 
            data = self.data[self.data.task_id.isin(self.return_tasks())]
            inputs =  torch.from_numpy(np.stack([eval(val) for val in data.input.values]))
            targets = torch.from_numpy(np.stack([0. if val=='A' else 1. for val in data.target.values]))
            
        self.inputs = inputs.reshape(self.batch_size, self.max_steps, self.num_dims)
        self.targets = targets.reshape(self.batch_size, self.max_steps).long()
        self.time = 0
        prev_targets = torch.randint_like(self.targets[:, self.time], low=0, high=self.num_choices)
        return self.inputs[:, self.time], self.targets[:, self.time], prev_targets, False, {'inputs':self.inputs, 'targets':self.targets}

    def step(self):
        """
        Take a step in the environment
        Returns:
            inputs: the inputs for the current step
            targets: the targets for the current step
            done: whether the episode is done
            info: additional information
        """
        done = False
        self.time += 1
        if self.time == (self.max_steps):
            return None, None, None, True, {}
        else:
            return self.inputs[:, self.time], self.targets[:, self.time], self.targets[:, self.time-1], done, {}

    def generate_synthetic_data(self, num_tasks=10000, split=[0.8, 0.1, 0.1]):
        
        self.num_tasks = num_tasks
        self.split = (torch.tensor([split[0], split[0]+split[1], split[0]+split[1]+split[2]]) * num_tasks).int()
        self.x = torch.randn(self.max_steps, self.num_tasks, self.num_dims)
        self.w = torch.randn(self.num_tasks, self.num_dims)
        self.c = torch.sigmoid((self.x * self.w).sum(-1)).round()

    def get_synthetic_data(self, mode=None):
        num_tasks = self.num_tasks
        tasks = np.arange(num_tasks)[:self.split[0]] if self.mode == 'train' else np.arange(num_tasks)[self.split[0]:self.split[1]] if self.mode == 'val' else np.arange(num_tasks)[self.split[1]:]
        ## randomize the order of the tasks
        #np.random.shuffle(tasks)
    
        # get batched data 
        mode = self.mode if mode is None else mode
        if mode == 'train':
            tasks = np.random.choice(tasks, self.batch_size, replace=False)
        elif mode == 'val':
            self.batch_size = self.split[1] - self.split[0]
        elif mode == 'test':
            self.batch_size = self.split[2] - self.split[1]
        inputs = self.x.permute(1, 0, 2)[tasks]
        targets = self.c.permute(1, 0)[tasks]

        return inputs, targets

    def sample_batch(self):

        data = self.data[self.data.task_id.isin(self.return_tasks())]
        # import ipdb ; ipdb.set_trace()
        # flip targets to 0 or 1 based on a random number
        data['target'] = data['target'].apply(lambda x: 0. if x=='A' else 1.) if torch.rand(1) > 0.5 else data['target'].apply(lambda x: 1. if x=='A' else 0.)
        data['input'] = data['input'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
        # shuffle the order of trials within a task but keep all the trials 
        if self.shuffle_trials:
            data = data.groupby('task_id').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
        # group all inputs for a task into a list
        data = data.groupby('task_id').agg({'input':list, 'target':list}).reset_index()
        # flip the target for %noise of total number of trials within each task
        if self.noise > 0.:
            data['target'] = data.groupby('task_id').target.apply(lambda x: x.sample(frac=self.noise).apply(lambda x: 1. if x==0. else 0.) if len(x) > 1 else x)
        # off set targets by 1 trial and randomly add zeros or ones in the beggining
        data['shifted_target'] = data['target'].apply(lambda x: [1. if torch.rand(1) > 0.5 else 0.] + x[:-1])
        stacked_task_features = [torch.from_numpy(np.concatenate((np.stack(task_input_features), np.stack(task_targets).reshape(-1, 1)),axis=1)) for task_input_features, task_targets in zip(data.input.values, data.shifted_target.values)]
        stacked_targets = [torch.from_numpy(np.stack(task_targets)) for task_targets in data.target.values]
        sequence_lengths = [len(task_input_features) for task_input_features in data.input.values]
        packed_inputs = rnn_utils.pad_sequence(stacked_task_features, batch_first=True)

        # import ipdb ; ipdb.set_trace() 
        # Split 'input' column into separate feature columns
        # data['input'] = data['input'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
        # data[['feature1', 'feature2', 'feature3']] = pd.DataFrame(data['input'].to_list(), index=data.index)
        
        # # group all input features for a task into a list
        # data = data.groupby('task_id').agg({'feature1':list, 'feature2':list, 'feature3':list, 'target':list}).reset_index()
        # sequence_lengths = data.feature1.apply(lambda x: len(x))
        # max_sequence_length = max(sequence_lengths)
        
        # convert lists of all features to numpy arrays and pad them to have same sequence length
        #data['feature1'] = data['feature1'].apply(lambda x: np.array(x)).apply(lambda x: np.pad(x, (0, max_sequence_length - len(x)), 'constant', constant_values=(0, 0)))
        #data['feature2'] = data['feature2'].apply(lambda x: np.array(x)).apply(lambda x: np.pad(x, (0, max_sequence_length - len(x)), 'constant', constant_values=(0, 0)))
        #data['feature3'] = data['feature3'].apply(lambda x: np.array(x)).apply(lambda x: np.pad(x, (0, max_sequence_length - len(x)), 'constant', constant_values=(0, 0)))
        #data['target'] = data['target'].apply(lambda x: np.array(x)).apply(lambda x: np.pad(x, (0, max_sequence_length - len(x)), 'constant', constant_values=(0, 0)))
        
        # pool all features from the dataframe into one array of dim (num_tasks, max_sequence_length, num_dims)
        #inputs = torch.from_numpy(np.stack([np.stack(data.feature1.values), np.stack(data.feature2.values), np.stack(data.feature3.values)], axis=2))
        #packed_inputs = rnn_utils.pad_sequence(stacked_list, batch_first=True) #, enforce_sorted=False)
        #rnn_utils.pack_sequence(inputs, enforce_sorted=False)

        return packed_inputs, sequence_lengths, stacked_targets #data.target.values

class ShepardsTask(nn.Module):
    """
    Categorisation task inspired by Shepard et al. (1961) for evaluating models on human performance
    """
    
    def __init__(self, max_steps=96, num_dims=3, batch_size=32, device='cpu', noise=0., shuffle_trials=False):
        super(ShepardsTask, self).__init__()
        self.device = torch.device(device)
        self.num_choices = 1 
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.noise = noise
        self.shuffle_trials = shuffle_trials

    def sample_batch(self, task_type=1):
  
        stacked_task_features, stacked_targets = self.generate_task(task_type)
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(stacked_task_features, batch_first=True)

        return packed_inputs, sequence_lengths, stacked_targets

    def generate_task(self, task_type):
        
        inputs_list, targets_list = [], []
        # generate all possible combinations of features
        all_feature_combinations = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0],\
                                                [0, 1, 1], [1, 0, 0], [1, 0, 1],\
                                                [1, 1, 0], [1, 1, 1]])
        
        for _ in range(self.batch_size):
            
            chosen_feature = np.random.choice(np.arange(3)) # choose one of the three feature dimensions randomly
            target = 0. if np.random.rand(1) > 0.5 else 1.

            # generate targets for each task type
            if task_type==1:
                # assign target for all objects with chosen feature == 0 and 1-target otherwise
                targets = np.array([target if feature_combination[chosen_feature]==0 else 1-target for feature_combination in all_feature_combinations])
            elif task_type==2:
                pass
            elif task_type==3:
                pass
            elif task_type==4:
                pass
            elif task_type==5:
                pass
            elif task_type==6:
                pass

            # add to the targets
            if self.noise > 0.:
                targets = np.array([target if np.random.rand(1) > self.noise else 1-target for target in targets])
            
            # concatenate all features and targets into one array with placed holder for shifted target
            concat_data = np.concatenate((all_feature_combinations, targets.reshape(-1, 1), targets.reshape(-1, 1)), axis=1)
            
            # create a new sampled data array sampling from the concatenated data array wih replacement
            sampled_data = concat_data[np.random.choice(np.arange(concat_data.shape[0]), self.max_steps, replace=True)]
            
            # replace placeholder with shifted targets to the sampled data array
            sampled_data[:, self.num_dims] = np.concatenate((np.array([0. if np.random.rand(1) > 0.5 else 1.]), sampled_data[:-1, self.num_dims]))
            
            # stacking all the sampled data across all tasks
            inputs_list.append(torch.from_numpy(sampled_data[:, :(self.num_dims+1)]))
            targets_list.append(torch.from_numpy(sampled_data[:, [self.num_dims+1]]))
    
        return inputs_list, targets_list
        
