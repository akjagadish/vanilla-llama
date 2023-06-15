import numpy as np
import pandas as pd
import torch.nn as nn
import torch

class CategorisationTask(nn.Module):
    """
    Categorisation task inspired by Shepard et al. (1961)
    """
    def __init__(self, data, max_steps=8, num_dims=3, batch_size=64, mode='train', split=[0.8, 0.1, 0.1]): 
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
        self.num_choices = self.data.target.nunique()
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.mode = mode
        self.split = (torch.tensor([split[0], split[0]+split[1], split[0]+split[1]+split[2]]) * self.data.task_id.nunique()).int()

    def return_tasks(self, mode=None):
        mode = self.mode if mode is None else mode
        if mode == 'train':
            tasks = np.random.choice(self.data.task_id.unique()[:self.split[0]], self.batch_size, replace=True)
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
        data = self.data[self.data.task_id.isin(self.return_tasks())]
        inputs = torch.from_numpy(np.stack([eval(val) for val in data.input.values]))
        targets = torch.from_numpy(np.stack([0. if val=='A' else 1. for val in data.target.values]))

        self.inputs = inputs.reshape(self.batch_size, self.max_steps, self.num_dims)
        self.targets = targets.reshape(self.batch_size, self.max_steps)
        self.time = 0
        
        return self.inputs[:, self.time], self.targets[:, self.time], False, {}

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
            return None, None, True, {}
        else:
            return self.inputs[:, self.time], self.targets[:, self.time], done, {}
