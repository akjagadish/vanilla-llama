import numpy as np
import pandas as pd

class CategorisationTask():
    metadata = {'render.modes': ['human']}
    def __init__(self, data, max_steps=8, num_dims=3, batch_size=64): 
        self.data = pd.read_csv(data)
        self.num_choices = self.data.target.nunique()
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_dims = num_dims

    def reset(self):
        tasks = np.random.choice(self.data.task_id.unique(), self.batch_size, replace=False)
        data = self.data[self.data.task_id.isin(tasks)]
        inputs = np.stack([eval(val) for val in data.input.values])
        targets = np.stack([val  for val in data.target.values])
        self.inputs = inputs.reshape(self.batch_size, self.max_steps, 3)
        self.targets = targets.reshape(self.batch_size, self.max_steps)
        self.time = 0
        

    def step(self):
        done = False
        self.time+= 1
        if self.time == self.max_steps:
            done = True
        return self.inputs[:, self.time], self.targets[:, self.time], done, {}
