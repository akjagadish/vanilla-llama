import pickle
import re
import pandas as pd
import numpy as np
import torch
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/data')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/rl2')
from baseline_classifiers import benchmark_baseline_models_regex_parsed_random_points, benchmark_baseline_models_regex_parsed
from baseline_classifiers import LogisticRegressionModel, SVMModel


def parse_generated_tasks(path, file_name, gpt, num_datapoints=8, last_task_id=0):
   
    # load llama generated tasks which were successfully regex parsed
    with open(f"{path}/{file_name}.txt", "rb") as fp:   
        datasets = pickle.load(fp)

    # regular expression pattern to extract input values from the stored inputs and targets
    pattern = r'((?: )?\s*[\d.]+)(?:,|;|\s)?\s*([\d.]+)(?:,|;|\s)?\s*([\d.]+)'

    # make a pandas dataframe for the parsed data
    df = None 
    task_id = last_task_id

    # parse the list using regex
    for task, data in enumerate(datasets):
        # initialize lists to store parsed values
        inputs, targets = [], []

        # load each input-target pair
        for item in data:
            if gpt == 'gpt3':
                #import ipdb; ipdb.set_trace()
                inputs.append([float(item[0]), float(item[1]), float(item[2])])
                targets.append(item[3])

            elif gpt == 'gpt4':
                #import ipdb; ipdb.set_trace()
                inputs.append([float(item[0]), float(item[1]), float(item[2])])
                targets.append(item[3][1] if len(item[3])>1 else item[3])
                
            else:
                match = re.match(pattern, item[0])
                #import ipdb; ipdb.set_trace()
                if match:
                    try:
                        inputs.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
                        targets.append(item[1])
                    except:
                        print(f'no match')
                else:
                    print(f'error parsing {task, item[0]}')

        # if the number of datapoints is equal to the number of inputs, add to dataframe
        if gpt=='gpt3' or gpt=='gpt4' or ((gpt=='llama') and (len(inputs)==num_datapoints)):
            print(f'inputs lengths {len(inputs)}')
            df = pd.DataFrame({'input': inputs, 'target': targets, 'trial_id': np.arange(len(inputs)), 'task_id': np.ones((len(inputs),))*(task_id)}) if df is None else pd.concat([df, \
                 pd.DataFrame({'input': inputs, 'target': targets, 'trial_id': np.arange(len(inputs)), 'task_id': np.ones((len(inputs),))*(task_id)})], ignore_index=True)
            task_id+=1
        else:
            print(f'dataset did not have {num_datapoints} datapoints but instead had {len(inputs)} datapoints')

    # save data frame to csv
    if df is not None:
        df.to_csv(f'{path}/{file_name}.csv')
    else:
        print(f'no datasets were successfully parsed')

    return task_id

def return_generated_task(path, gpt, model, num_dim, num_data, num_tasks, run, proc_id):
    return pd.read_csv(f"{path}/{gpt}_generated_tasks_params{model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}.csv")

def pool_generated_tasks(path, models, dims, data, tasks, runs, proc_ids):
    ''' 
    Pool the generated tasks from different processes into a single dataframe
    Args:
        path: path to the folder containing the generated tasks
        models: list of models used to generate the tasks
        dims: list of dimensions used to generate the tasks
        data: list of number of datapoints used to generate the tasks
        tasks: list of number of tasks used to generate the tasks
        runs: list of runs used to generate the tasks
        proc_ids: dictionary of process ids used to generate the tasks
    Returns:
        df: dataframe containing the pooled generated tasks
    '''

    df = None
    total_tasks = 0
    for model in models:
        for dim in dims:
            for num_data in data:
                for num_tasks in tasks:
                    for run in runs:
                        for proc_id in proc_ids[num_tasks]:
                            total_tasks += num_tasks
                            df = return_generated_task(path, model, dim, num_data, num_tasks, run, proc_id) if df is None else pd.concat([df, \
                                return_generated_task(path, model, dim, num_data, num_tasks, run, proc_id)], ignore_index=True)
              
                # save the pooled dataframe to csv
                df = df.query('target == "A" or target == "B"')
                #df['task_id'] = np.int64(np.arange(len(df))/num_data) #+ 1 
                df.to_csv(f"{path}/llama_generated_tasks_params{model}_dim{dim}_data{num_data}_tasks{total_tasks}.csv")
    
    return df

def return_baseline_performance(data, random=False):

    num_tasks = data.task_id.max()+1
    llm_performance = benchmark_baseline_models_regex_parsed(num_tasks, data)
    if random:
        uniform_performance = benchmark_baseline_models_regex_parsed_random_points(data)
        performance = np.concatenate((llm_performance, uniform_performance), axis=1)
    else:
        performance = llm_performance

    means = performance.mean(0)
    std_errors = performance.std(0)/np.sqrt(num_tasks-1)

    return means, std_errors, performance

# computing the distance between consequetive datapoints over trials
def l2_distance_trials(data, within_targets=False, within_consecutive_targets=False):
    '''
    Spatial distance between datapoints for each task
    Args:
        data: pandas dataframe with columns ['task_id', 'trial_id', 'input', 'target']
    Returns:
        None
    ''' 
    tasks = data.task_id.unique()#[:100]
    # extract the spatial distance for each task

    for target in data.target.unique():

        for task in tasks:
            # get the inputs for this task which is numpy array of dim (num_trials, 3)
            inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
            # get the targets for this task which is numpy array of dim (num_trials, 1)
            targets = np.stack([val for val in data[data.task_id==task].target.values])

        
            if within_targets:
                inputs = inputs[targets==target]    

            # get the spatial distance between datapoints over trials for only points with the same target
            distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[ii+1,:]) for ii in range(inputs.shape[0]-1)])
            
            if within_consecutive_targets:
                # consequetive datapoints with the same target
                distance = np.array([np.linalg.norm(inputs[ii]-inputs[ii+1]) for ii in range(inputs.shape[0]-1) if targets[ii]==targets[ii+1]])
            
            # pad with Nan's if distances are of unequal length and stack them vertically over tasks
            distance = np.pad(distance, (0, int(data.trial_id.max()*0.6)-distance.shape[0] if within_targets else data.trial_id.max()-distance.shape[0]), mode='constant', constant_values=np.nan)
            if task==0:
                distances = distance
            else:
                distances = np.vstack((distances, distance))

        # # plot the spatial distances
        # f, ax = plt.subplots(1, 1, figsize=(7,7))   
        # sns.heatmap(distances, annot=False, ax=ax, cmap='hot_r', vmin=0, vmax=1)
        # ax.set_title(f'Distance between datapoints')
        # ax.set_xlabel('Trial')
        # ax.set_ylabel('Task')
        # plt.show()
    
    return distances

def l2_distance_trials_all(data, target='A', shift=1, within_targets=False, llama=False, random=False):
    '''
    Compute distance of a datapoint with every other datapoint with shifts over trials
    Args:
        data: pandas dataframe with columns ['task_id', 'trial_id', 'input', 'target']
    Returns:
        None
    ''' 
    tasks = data.task_id.unique()#[:1000]

    # extract the distances for each task
    for task in tasks:
        
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        targets = np.stack([val for val in data[data.task_id==task].target.values]) 
        
        if random:
            num_points, dim = 87, 3
            inputs =  np.random.rand(num_points, dim) 
            targets = np.random.choice(['A', 'B'], size=num_points) 

        if within_targets:
            inputs = inputs[targets==target]    

        # get the spatial distance between datapoints over trials 
        distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[ii+shift,:]) for ii in range(inputs.shape[0]-shift)])
        # pad with Nan's if distances are of unequal length and stack them vertically over tasks
        if llama:
            distance = np.pad(distance, (0, int(8*0.6)-distance.shape[0] if within_targets else 8-distance.shape[0]), mode='constant', constant_values=np.nan)
        else:
            distance = np.pad(distance, (0, int(data.trial_id.max()*0.9)-distance.shape[0] if within_targets else data.trial_id.max()+1-distance.shape[0]), mode='constant', constant_values=np.nan)

        if task==0:
            distances = distance
        else:
            distances = np.vstack((distances, distance))
        
    return distances

def probability_same_target_vs_distance(data, target='A', llama=False, random=False):

    #TODO:
    # 1. set max datapoints based on max number of trials in the dataset
    # 2. set values for random more appropriately

    tasks = data.task_id.unique()#[:1000]
    MAX_SIZE = (data.trial_id.max()+1)**2
    # load data for each task
    for task in tasks:
        
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        targets = np.stack([val for val in data[data.task_id==task].target.values]) 
        
        if random:
            num_points, dim = data.trial_id.max(), 3
            inputs =  np.random.rand(num_points, dim) 
            targets = np.random.choice(['A', 'B'], size=num_points) 
   

        # get the spatial distance between datapoints over trials 
        #feature_distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[ii+1,:]) for ii in range(inputs.shape[0]-1)])
        # distance between every pair of datapoints
        feature_distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[jj,:]) for ii in range(inputs.shape[0]) for jj in range(inputs.shape[0]) if ii!=jj])

        # compute difference in probability of target for each pair of datapoints
        svm = SVMModel(inputs, targets)
        probability = svm.predict_proba(inputs)
        #probability_distance = np.array([np.linalg.norm(probability[ii, 0]-probability[ii+1, 0]) for ii in range(probability.shape[0]-1)])
        probability_distance = np.array([np.linalg.norm(probability[ii,0]-probability[jj,0]) for ii in range(probability.shape[0]) for jj in range(probability.shape[0]) if ii!=jj])
        
        # pad with Nan's if distances are of unequal length and stack them vertically over tasks
        # 100*100 = 10000 is the maximum number of pairs of datapoints as number of datapoints is 100
        probability_distance = np.pad(probability_distance, (0, MAX_SIZE-feature_distance.shape[0]), mode='constant', constant_values=np.nan)
        feature_distance = np.pad(feature_distance, (0, MAX_SIZE-feature_distance.shape[0]), mode='constant', constant_values=np.nan)
        
        #print(probability_distance.shape)
        if task==0:
            distances = feature_distance
            probabilities = probability_distance
        else:
            distances = np.vstack((distances, feature_distance))
            probabilities = np.vstack((probabilities, probability_distance))

    # # plot probability vs distance
    # f, ax = plt.subplots(1, 1, figsize=(7,7))
    # sns.regplot(distances.flatten(), probabilities.flatten(), ax=ax)
    # ax.set_title(f'Probability of same target vs distance between datapoints')
    # ax.set_xlabel('Distance between datapoints')
    # ax.set_ylabel('Probability of same target')
    # plt.show()

    return distances, probabilities

def evaluate_data_against_baselines(data, upto_trial=15, return_all=True):

    tasks = data.task_id.unique()#[:1000] 
    accuracy_lm = []
    accuracy_svm = []
    # loop over dataset making predictions for next trial using model trained on all previous trials
    for task in tasks:
        baseline_model_choices, true_choices = [], []   
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        targets = torch.stack([torch.tensor(0) if val=='A' else torch.tensor(1) for val in data[data.task_id==task].target.values])
        num_trials = data[data.task_id==task].trial_id.max()

        trial = upto_trial # fit datapoints upto upto_trial; sort of burn-in trials
        # loop over trials
        while trial <= num_trials:
            trial_inputs = inputs[:trial]
            trial_targets = targets[:trial]
            try:
                lr_model = LogisticRegressionModel(trial_inputs, trial_targets)
                svm_model = SVMModel(trial_inputs, trial_targets)
                lr_model_choice = lr_model.predict_proba(inputs[trial:trial+1])
                svm_model_choice = svm_model.predict_proba(inputs[trial:trial+1])
                true_choice = targets[trial:trial+1]
                baseline_model_choices.append(torch.tensor([lr_model_choice, svm_model_choice]))
                true_choices.append(true_choice)
            except:
                print('error fitting : for example not enough datapoints for a class')
            trial += 1
    
        # calculate accuracy
        baseline_model_choices_stacked, true_choices_stacked = torch.stack(baseline_model_choices).squeeze().argmax(2), torch.stack(true_choices).squeeze()
        accuracy_per_task_lm = (baseline_model_choices_stacked[:, 0] == true_choices_stacked) #for model_id in range(1)]
        accuracy_per_task_svm = (baseline_model_choices_stacked[:, 1] == true_choices_stacked) #for model_id in range(1)]
        
        accuracy_lm.append(accuracy_per_task_lm)
        accuracy_svm.append(accuracy_per_task_svm)
        
    return accuracy_lm, accuracy_svm

def find_counts(inputs, dim, xx_min, xx_max):
    return (inputs[:, dim]<xx_max)*(inputs[:, dim]>xx_min)

def data_in_range(inputs, targets, min_value=0, max_value=1):
    inputs_in_range = [(inputs[ii]>min_value).all() * (inputs[ii]<max_value).all() for ii in range(len(inputs))]
    inputs = inputs[inputs_in_range]
    targets = targets[inputs_in_range]
    return inputs, targets

def bin_data_points(num_bins, data, min_value=0, max_value=1):
    inputs = np.stack([eval(val) for val in data.input.values])
    targets = np.stack([val for val in data.target.values])
    inputs, targets = data_in_range(inputs, targets, min_value, max_value)
    bins = np.linspace(0, 1, num_bins+1)[:-1]
    bin_counts, target_counts = [], [] #np.zeros((len(bins)*3))
    for ii in bins:
        x_min = ii 
        x_max = ii + 1/num_bins
        for jj in bins:
            y_min = jj
            y_max = jj + 1/num_bins
            for kk in bins:
                z_min = kk
                z_max = kk + 1/num_bins
                num_points = (find_counts(inputs, 0, x_min, x_max)*find_counts(inputs, 1, y_min, y_max)*find_counts(inputs, 2, z_min, z_max))
                bin_counts.append(num_points.sum())
                target_counts.append((targets[num_points]=='A').sum())

    bin_counts = np.array(bin_counts)
    target_counts = np.array(target_counts)
    return 

def return_data_stats(data):

    df = data.copy()
    max_tasks = int(df['task_id'].max() + 1)
    all_corr, all_coef, all_bics_linear, all_bics_quadratic  = [], [], [], []
    for i in range(0, max_tasks):
        df_task = df[df['task_id'] == i]
        if len(df_task) > 40: # arbitary data size threshold
            y = df_task['target'].to_numpy()
            y = np.unique(y, return_inverse=True)[1]

            df_task['input'] = df_task['input'].apply(eval).apply(np.array)
            X = df_task["input"].to_numpy()
            X = np.stack(X)
            
            # correlations
            all_corr.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
            all_corr.append(np.corrcoef(X[:, 0], X[:, 2])[0, 1])
            all_corr.append(np.corrcoef(X[:, 1], X[:, 2])[0, 1])


            if (y == 0).all() or (y == 1).all():
                pass
            else:
                X_linear = PolynomialFeatures(1).fit_transform(X)
                log_reg = sm.Logit(y, X_linear).fit(method='bfgs')

                # weights
                all_coef.append(log_reg.params[1])
                all_coef.append(log_reg.params[2])
                all_coef.append(log_reg.params[3])

                X_poly = PolynomialFeatures(2).fit_transform(X)
                log_reg_quadratic = sm.Logit(y, X_poly).fit(method='bfgs')

                # bics
                all_bics_linear.append(log_reg.bic)
                all_bics_quadratic.append(log_reg_quadratic.bic)

    # compute posterior probabilities
    logprobs = torch.from_numpy(-0.5 * np.stack((all_bics_linear, all_bics_quadratic), -1))
    joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[1])
    marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
    posterior_logprob = joint_logprob - marginal_logprob

    return all_corr, all_coef, posterior_logprob