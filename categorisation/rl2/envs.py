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

        return packed_inputs, sequence_lengths, stacked_targets 
    
class SyntheticCategorisationTask(nn.Module):
    """
    Generate synthetic data for the Categorisation task inspired by Shepard et al. (1961)
    """
    def __init__(self, max_steps=8, num_dims=3, batch_size=64, mode='train', split=[0.8, 0.1, 0.1], device='cpu', num_tasks=10000, noise=0.1, shuffle_trials=False): 
        """ 
        Initialise the environment
        Args: 
            data: path to csv file containing data
            max_steps: number of steps in each episode
            num_dims: number of dimensions in each input
            batch_size: number of tasks in each batch
        """
        super(SyntheticCategorisationTask, self).__init__()

        self.device = torch.device(device)
        self.num_tasks = num_tasks
        self.num_choices = 1 
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.mode = mode
        self.split = (torch.tensor([split[0], split[0]+split[1], split[0]+split[1]+split[2]]) * self.num_tasks).int()
        self.noise = noise
        self.shuffle_trials = shuffle_trials
        self.generate_synthetic_data()

    def generate_synthetic_data(self):
    
        self.x = torch.randn(self.max_steps, self.num_tasks, self.num_dims)
        self.w = torch.randn(self.num_tasks, self.num_dims)
        self.c = torch.sigmoid((self.x * self.w).sum(-1)).round()

    def get_synthetic_data(self, mode=None):
        
        num_tasks = self.num_tasks
        tasks = np.arange(num_tasks)[:self.split[0]] if self.mode == 'train' else np.arange(num_tasks)[self.split[0]:self.split[1]] if self.mode == 'val' else np.arange(num_tasks)[self.split[1]:]
        
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

    def sample_batch(self, mode='train'):
        
        # generate synthetic data
        inputs, targets = self.get_synthetic_data(mode)
        # shuffle the order of trials within a task but keep all the trials 
        if self.shuffle_trials:
            permutation = torch.randperm(inputs.shape[1])
            inputs, targets = inputs[:, permutation], targets[:, permutation]
        # flip the target for %noise of total number of trials within each task
        if self.noise > 0.:
            targets = torch.stack([target if torch.rand(1) > self.noise else 1-target for target in targets])
        # off set targets by 1 trial and randomly add zeros or ones in the beggining
        shifted_targets = torch.stack([torch.cat((torch.tensor([1. if torch.rand(1) > 0.5 else 0.]), target[:-1])) for target in targets])
        # stacking input and targets
        stacked_task_features = torch.cat((inputs, shifted_targets.unsqueeze(2)), dim=2)
        stacked_targets = targets
        sequence_lengths = [len(task_input_features) for task_input_features in inputs]
        packed_inputs = rnn_utils.pad_sequence(stacked_task_features, batch_first=True)

        return packed_inputs.to(self.device), sequence_lengths, stacked_targets.to(self.device) 

class ShepardsTask(nn.Module):
    """
    Categorisation task inspired by Shepard et al. (1961) for evaluating models on human performance
    """
    
    def __init__(self, task=None, max_steps=96, num_dims=3, batch_size=64, device='cpu', noise=0., shuffle_trials=False):
        super(ShepardsTask, self).__init__()
        self.device = torch.device(device)
        self.num_choices = 1 
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.noise = noise
        self.shuffle_trials = shuffle_trials
        self.task_type = task

    def sample_batch(self, task_type=1):
        task_type = self.task_type if self.task_type is not None else task_type
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
            
            # generate a random target
            target = 0. if np.random.rand(1) > 0.5 else 1.
            # generate targets for each task type
            if task_type==1:
                chosen_feature = np.random.choice(np.arange(3)) # choose one of the three feature dimensions randomly
                # assign target for all objects with chosen feature == 0 and 1-target otherwise
                targets = np.array([target if feature_combination[chosen_feature]==0 else 1-target for feature_combination in all_feature_combinations])
            
            elif task_type==2:
                chosen_features = np.random.choice(np.arange(3), 2, replace=False) # choose two of the three feature dimensions randomly
                # assign target when the values for these two chosen feature are the same as 0 and 1-target otherwise
                targets = np.array([target if feature_combination[chosen_features[0]]==feature_combination[chosen_features[1]] else 1-target for feature_combination in all_feature_combinations])
                
            elif task_type==3:
                np.random.shuffle(all_feature_combinations) # shuffle rows in all_feature_combinations
                chosen_feature = np.random.choice(np.arange(3)) # choose one of the three feature dimensions randomly
                category_indices = np.hstack((np.where(all_feature_combinations[:, chosen_feature]==1)[0][:3], np.where(all_feature_combinations[:, chosen_feature]==0)[0][0]))
                # category_2_indices = np.hstack((np.where(all_feature_combinations[:, chosen_feature]==1)[0][3], np.where(all_feature_combinations[:, chosen_feature]==0)[0][1:]))
                # assign target to category 1 indices and 1-target to category 2 indices
                targets = np.array([target if i in category_indices else 1-target for i in range(len(all_feature_combinations))])

            elif task_type==4:
                # choose one arbitrary instance as the prototype from all feature combinations
                prototype = all_feature_combinations[np.random.choice(np.arange(len(all_feature_combinations)))]
                # assign target to instances which have at least two features in common with the prototype and 1-target otherwise
                targets = np.array([target if np.sum(prototype==feature_combination)>=2 else 1-target for feature_combination in all_feature_combinations])
                
            elif task_type==5:
                np.random.shuffle(all_feature_combinations) # shuffle rows in all_feature_combinations
                chosen_feature = np.random.choice(np.arange(3)) # choose one of the three feature dimensions randomly
                category_indices = np.hstack((np.where(all_feature_combinations[:, chosen_feature]==1)[0][:2], np.where(all_feature_combinations[:, chosen_feature]==0)[0][:2]))
                # assign target to category 1 indices and 1-target to category 2 indices
                targets = np.array([target if i in category_indices else 1-target for i in range(len(all_feature_combinations))])

            elif task_type==6:
                # choose one arbitrary instance as the prototype from all feature combinations
                prototype = all_feature_combinations[np.random.choice(np.arange(len(all_feature_combinations)))]
                # assign target to instances which have at least two features in common with the prototype and 1-target otherwise
                targets = np.array([target if np.sum(prototype==feature_combination)==1 or np.sum(prototype==feature_combination)==3 else 1-target for feature_combination in all_feature_combinations])

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
        

class NosofskysTask(nn.Module):

    def __init__(self, task=[4, None, None], num_blocks=1, num_dims=3, batch_size=64, device='cpu'):
        super(NosofskysTask, self).__init__()
        self.device = torch.device(device)
        self.num_choices = 1 
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.task = task
        self.num_blocks = num_blocks
        # experimental input: value (brightness)/chroma (saturation)
        features = np.array([[7, 4], [7, 8], [6, 6], [6, 10], [5, 4], [5, 8], [5, 12], [4, 6], [4, 10], [3, 4], [3, 8], [3, 10]])
        #self.input = self.input/10 # normalise the input to be between 0 and 1
        self.input = (features-features.min(0))
        self.input = self.input/self.input.max(0)
        #TODO: change the dimension I concatenate zeros everyime 
        #TODO: use 0.5 instead of 0 for non feature
        # concate zeros at different points to the input to make it 3 dimensions
        self.input = np.concatenate((np.zeros((self.input.shape[0], 1)), self.input), axis=1)  #np.concatenate((self.input, np.zeros((self.input.shape[0], 1))), axis=1) 
        self.input = self.input[:, np.random.permutation(self.input.shape[1])] #input[:, [0, 1, 2]] #
        self.target = np.array([0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0., 0.])
        self.instance_labels = np.arange(len(self.input))

    def sample_batch(self):
        stacked_task_features, stacked_targets, stacked_labels = self.generate_task()
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(stacked_task_features, batch_first=True)
        self.stacked_labels = stacked_labels
        
        return packed_inputs, sequence_lengths, stacked_targets
    
    def generate_task(self):

        inputs_list, targets_list, labels_list = [], [], []
        
        num_repeats, instance, num_instance = self.task
        for _ in range(self.batch_size):

            # repeat all inputs except the instance for num_repeats times and repeat the instance for num_instance times
            if instance is None:
                inputs = np.repeat(self.input, num_repeats, axis=0)
                targets = np.repeat(self.target, num_repeats, axis=0)
                instance_labels = np.repeat(self.instance_labels, num_repeats, axis=0)
            else:
                inputs = np.repeat(self.input[np.arange(len(self.input))!=instance], num_repeats, axis=0)
                targets = np.repeat(self.target[np.arange(len(self.target))!=instance], num_repeats, axis=0)
                instance_labels = np.repeat(self.instance_labels[np.arange(len(self.instance_labels))!=instance], num_repeats, axis=0)
                inputs = np.concatenate((inputs, np.repeat([self.input[instance]], num_instance*num_repeats, axis=0)), axis=0)
                targets = np.concatenate((targets, np.repeat(self.target[instance], num_instance*num_repeats, axis=0)), axis=0)
                instance_labels = np.concatenate((instance_labels, np.repeat(self.instance_labels[instance], num_instance*num_repeats, axis=0)), axis=0)

            # concatenate all features and targets into one array with placeholder for shifted targets
            concat_data = np.concatenate((inputs, targets.reshape(-1, 1), targets.reshape(-1, 1), instance_labels.reshape(-1,1)), axis=1)
            
            # shuffle the data differently in every iteration
            np.random.shuffle(concat_data)

            # replace placeholder with shifted target with the targets
            concat_data[:, self.num_dims] = np.concatenate((np.array([0. if np.random.rand(1) > 0.5 else 1.]), concat_data[:-1, self.num_dims]))

            # repeat all blocks for num_blocks times
            #concat_data = np.repeat(concat_data, self.num_blocks, axis=0)
            # print(concat_data.shape)
            # shuffle the data differently in every iteration
            # np.random.shuffle(concat_data)

            # stacking all the sampled data across all tasks
            inputs_list.append(torch.from_numpy(concat_data[:, :(self.num_dims+1)]))
            targets_list.append(torch.from_numpy(concat_data[:, [self.num_dims+1]]))
            labels_list.append(torch.from_numpy(concat_data[:, [self.num_dims+2]]))

        return inputs_list, targets_list, labels_list
    
class LeveringsTask(nn.Module):
    """
    Categorisation task inspired by Levering et al. (2019) for evaluating meta-learned model 
    on linear and non-linear decision boundaries
    """
    
    def __init__(self, task='linear', max_steps=158, num_blocks=25, num_train=6, num_dims=3, batch_size=64, device='cpu', noise=0., shuffle_trials=False):
        super(LeveringsTask, self).__init__()
        self.device = torch.device(device)
        self.num_choices = 1 
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.noise = noise
        self.shuffle_trials = shuffle_trials
        self.num_blocks = num_blocks
        self.num_train = num_train
        # generate all possible combinations of features
        self.input = np.array([[0, 0, 1], [0, 1, 0],\
                                [0, 1, 1], [1, 0, 0], [1, 0, 1],\
                                [1, 1, 0], [0, 0, 0], [1, 1, 1]])
        self.target = np.array([0., 0., 1., 0., 1., 1., 0., 1.]) if task=='linear' else np.array([0., 1., 0., 0., 1., 1., 0., 1.])

    def sample_batch(self):
     
        stacked_task_features, stacked_targets = self.generate_task()
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(stacked_task_features, batch_first=True)

        return packed_inputs, sequence_lengths, stacked_targets

    def generate_task(self):
        
        inputs_list, targets_list = [], []
        
        for _ in range(self.batch_size):
            
            targets = self.target if torch.rand(1) > 0.5 else 1-self.target  # flip self.target to 0 or 1 based on a random number
            targets = targets[:self.num_train]
            inputs = self.input[:self.num_train]

            # repeat all inputs and targets for num_blocks times
            inputs = np.repeat(inputs, self.num_blocks, axis=0)
            targets = np.repeat(targets, self.num_blocks, axis=0)

            # add noise to the targets
            if self.noise > 0.:
                targets = np.array([target if np.random.rand(1) > self.noise else 1-target for target in targets])
            
            # concatenate all features and targets into one array with placed holder for shifted target
            concat_data = np.concatenate((inputs, targets.reshape(-1, 1), targets.reshape(-1, 1)), axis=1)

            # shuffle the data differently in every iteration
            np.random.shuffle(concat_data)

            # replace placeholder with shifted target with the targets
            concat_data[:, self.num_dims] = np.concatenate((np.array([0. if np.random.rand(1) > 0.5 else 1.]), concat_data[:-1, self.num_dims]))

            # make evaluation block at the end of the training block
            #TODO: not giving correct target as inputs
            # eval_data = np.concatenate((self.input, self.target.reshape(-1, 1), self.target.reshape(-1, 1)), axis=1)
            # np.random.shuffle(eval_data)
            # eval_data[:, self.num_dims] = np.concatenate((np.array([0. if np.random.rand(1) > 0.5 else 1.]), eval_data[:-1, self.num_dims]))

            # stack concat_data and eval_data
            data = concat_data #np.vstack((concat_data, eval_data))

            # stacking all the sampled data across all tasks
            inputs_list.append(torch.from_numpy(data[:, :(self.num_dims+1)]))
            targets_list.append(torch.from_numpy(data[:, [self.num_dims+1]]))
    
        return inputs_list, targets_list            