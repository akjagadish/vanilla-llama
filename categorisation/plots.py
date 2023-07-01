import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/rl2')
from utils import return_baseline_performance

# set plotting parameters
COLORS = {'compositional':'#117733', 
          'noncompositional':'#96CAA7',
          'lr':'#88CCEE',#882255
          'svm':'#CC6677',
          'optimal': '#D6BF4D',
          'mean_tracker':'#882255', #332288
          'mean_tracker_compositional':'#882255', #AA4499',
          'rbf_nocontext_nomemory':'#44AA99', 
          'simple_grammar_constrained':'#44AA99',
          'simple_grammar_constrained_noncompositonal':'#EF9EBB',
          'rl2':'#E2C294', #'#0571D0', 
          'metal':'#DA9138', #"#D55E00", 
          }
FONTSIZE=20

def compare_llm_uniform_data_samples(data, random=False):

    means, std_errors, performance = return_baseline_performance(data, random=random)
    conditions =  ['LR', 'SVM', 'Uni', 'Uni'] if random else ['LR', 'SVM']
    colors =  [COLORS['lr'], COLORS['svm']] if random else [COLORS['lr'], COLORS['svm']]
    FIGSIZE=(5,5)

    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ## bar plot
    ax.bar(np.arange(0,len(means)), means, color=colors, label=conditions, width=.9)
    ax.errorbar(np.arange(0,len(means)), means, yerr=std_errors, color='r', lw=3, fmt='o') #linestyle='solid')

    ## plot individual points
    w = 0.5
    ax.hlines(0.5, -0.5, len(means)-0.5, color='k', linestyles='dotted', lw=5)#, label='Random')
    for i in range(len(means)):
        # distribute scatter randomly across whole width of bar
        ax.scatter(i + np.random.random(performance.shape[0]) * w - w / 2, performance[:, i], color='k', alpha=.3, zorder=3)

    ## formatting    
    plt.xticks(np.arange(0,len(means)))
    plt.yticks(fontsize=FONTSIZE-2)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel(f'Score', fontsize=FONTSIZE)#$a_{trials}$
    ax.set_xticklabels(conditions, fontsize=FONTSIZE-4)#['', '']
    sns.despine()
    f.tight_layout()
    plt.show()

def label_imbalance(data, categories=['A','B']):

    num_tasks = int(data.task_id.max()+1)
    num_targets = np.stack([(data[data.task_id==task_id].target=='A').sum() for task_id in data.task_id.unique()])
    expected_number_points = np.array([data[data.task_id==ii].trial_id.max()+1 for ii in np.arange(num_tasks)]).mean()

    f, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.bar(categories, [num_targets.mean(), expected_number_points-num_targets.mean()])
    ax.errorbar(categories, [num_targets.mean(), expected_number_points-num_targets.mean()], yerr=[num_targets.std(), num_targets.std()], c='k')
    #plt.legend(fontsize=FONTSIZE-5,  loc="upper center", bbox_to_anchor=(.45, 1.1), ncol=3, frameon=True)
    ax.set_ylabel('Mean number of points per class', fontsize=FONTSIZE-2)
    ax.set_xlabel('Category', fontsize=FONTSIZE-2) #$a_{name_trials}$
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()


# plotting the autocorrelations between features
def plot_autocorr_features(data):
    '''
    Autocorrelation between features for each task
    Args:
        data: pandas dataframe with columns ['task_id', 'trial_id', 'input', 'target']
    Returns:
        None
    '''
    tasks = data.task_id.unique()
    # extract the autocorrelations for each task
    for task in tasks:
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        targets = np.stack([val for val in data[data.task_id==task].target.values])
        
        # get the autocorrelations between input dimensions over trials
        autocorr = np.array([np.corrcoef(inputs[:,ii], inputs[:,jj])[0,1] for ii in range(inputs.shape[1]) for jj in range(inputs.shape[1])])
        # reshape the autocorrelations to be of dim (num_trials, num_input_dims, num_input_dims)
        autocorr = autocorr.reshape((inputs.shape[1], inputs.shape[1]))
        
        # plot the autocorrelations
        f, ax = plt.subplots(1, 1, figsize=(7,7))
        sns.heatmap(autocorr, annot=True, ax=ax, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Task {task}')
        plt.show()

# plotting the distance between datapoints over trials
def l2_distance_trials(data,within_targets=False, within_consquetive_targets=False):
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
            
            if within_consquetive_targets:
                # consequetive datapoints with the same target
                distance = np.array([np.linalg.norm(inputs[ii]-inputs[ii+1]) for ii in range(inputs.shape[0]-1) if targets[ii]==targets[ii+1]])
            
            # pad with Nan's if distances are of unequal length and stack them vertically over tasks
            distance = np.pad(distance, (0, int(data.trial_id.max()*0.6)-distance.shape[0] if within_targets else data.trial_id.max()-distance.shape[0]), mode='constant', constant_values=np.nan)
            if task==0:
                distances = distance
            else:
                distances = np.vstack((distances, distance))
        
        # plot the spatial distances
        f, ax = plt.subplots(1, 1, figsize=(7,7))   
        sns.heatmap(distances, annot=False, ax=ax, cmap='hot_r', vmin=0, vmax=1)
        ax.set_title(f'Distance between datapoints')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Task')
        plt.show()
    
    return distances

