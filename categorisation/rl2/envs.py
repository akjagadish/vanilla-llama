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
    def __init__(self, data, max_steps=8, num_dims=3, batch_size=64, mode='train', split=[0.8, 0.1, 0.1], device='cpu', synthetic_data=False, num_tasks=10000): 
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
        # covert targets to 0 and 1 but switch them based on random number
        random_number = torch.rand(1)
        data['target'] = data['target'].apply(lambda x: 0. if x=='A' else 1.) if random_number > 0.5 else data['target'].apply(lambda x: 1. if x=='A' else 0.)
        data['input'] = data['input'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
        # shuffle the order of trials within a task but keep all trials 
        data = data.groupby('task_id').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
        # group all inputs for a task into a list
        data = data.groupby('task_id').agg({'input':list, 'target':list}).reset_index()
        # off set targets by 1 trial but zeros in the beggining
        data['shifted_target'] = data['target'].apply(lambda x: [0. if random_number > 0.5 else 1.] + x[:-1])
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

class HumanCategorizationTask(nn.Module):
    """
    Categorisation task inspired by Shepard et al. (1961) for evaluating models on human performance
    """
    
    def __init__(self, data, max_steps=96, num_dims=3, batch_size=32, device='cpu'):
        super(HumanCategorizationTask, self).__init__()
        self.device = torch.device(device)
        self.num_choices = 1 
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_dims = num_dims

    def generate_tasks(self):
        pass
