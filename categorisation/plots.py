import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/rl2')
from utils import return_baseline_performance, return_data_stats
from baseline_classifiers import LogisticRegressionModel, SVMModel
from evaluate import evaluate_1d
from utils import evaluate_data_against_baselines, bin_data_points
from utils import probability_same_target_vs_distance

# set plotting parameters
COLORS = {'a':'#117733', 
          'b':'#96CAA7',
          'lr':'#88CCEE',#882255
          'svm':'#CC6677',
          'optimal': '#D6BF4D',
          'feature_1':'#882255', #332288
          'mean_tracker_compositional':'#882255', #AA4499',
          'stats':'#44AA99', 
          'feature_2':'#44AA99',
          'simple_grammar_constrained_noncompositonal':'#EF9EBB',
          'feature_3':'#E2C294', #'#0571D0', 
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
    plt.xticks(np.arange(0,len(means)), fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel(f'Accuracy', fontsize=FONTSIZE)#$a_{trials}$
    ax.set_xticklabels(conditions, fontsize=FONTSIZE-2)#['', '']
    sns.despine()
    f.tight_layout()
    plt.show()

def label_imbalance(data, categories=['A','B']):

    num_tasks = int(data.task_id.max()+1)
    num_targets = np.stack([(data[data.task_id==task_id].target=='A').sum() for task_id in data.task_id.unique()])
    expected_number_points = np.nanmean(np.array([data[data.task_id==ii].trial_id.max()+1 for ii in np.arange(num_tasks)]))
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.bar(categories, [num_targets.mean(), expected_number_points-num_targets.mean()], color=[COLORS['a'], COLORS['b']])
    ax.errorbar(categories, [num_targets.mean(), expected_number_points-num_targets.mean()], yerr=[num_targets.std(), num_targets.std()], c='k')
    #plt.legend(fontsize=FONTSIZE-5,  loc="upper center", bbox_to_anchor=(.45, 1.1), ncol=3, frameon=True)
    ax.set_ylabel('Mean number of points per class', fontsize=FONTSIZE)
    ax.set_xlabel('Category', fontsize=FONTSIZE) #$a_{name_trials}$
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()


# plot mean number of tasks
def plot_mean_number_tasks(data):
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    expected_number_points = np.array([data[data.task_id==ii].trial_id.max()+1 for ii in np.arange(data.task_id.max()+1)])
    print('mean: ', expected_number_points.mean())
    ax.hist(expected_number_points)
    plt.legend(fontsize=FONTSIZE-2,  loc="upper center", bbox_to_anchor=(.45, 1.1), ncol=3, frameon=False)
    ax.set_ylabel('Counts', fontsize=FONTSIZE)
    ax.set_xlabel('Number of data points per task', fontsize=FONTSIZE) #$a_{name_trials}$
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
    # store the autocorrelations for each task
    autocorrs = []

    # extract the autocorrelations for each task
    for task in tasks:
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        
        # get the autocorrelations between input dimensions over trials
        autocorr = np.array([np.corrcoef(inputs[:,ii], inputs[:,jj])[0,1] for ii in range(inputs.shape[1]) for jj in range(inputs.shape[1])])
        # reshape the autocorrelations to be of dim (num_trials, num_input_dims, num_input_dims)
        autocorr = autocorr.reshape((inputs.shape[1], inputs.shape[1]))
        
        # store the autocorrelations for this task
        autocorrs.append(autocorr)
    
    # plot the mean autocorrelations 
    f, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.imshow(np.stack(autocorrs).mean(axis=0), cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(0,3))
    ax.set_yticks(np.arange(0,3))
    ax.set_xticklabels(['x1', 'x2', 'x3'])
    ax.set_title(f'Task {task}')
    plt.show()

    return autocorrs

# plot the correlation between each feature over time
def plot_correlation_features(data, max_input_length=100, num_features=3, time_shift=1):
    '''
    Correlation between features for each task
    Args:
        data: pandas dataframe with columns ['task_id', 'trial_id', 'input', 'target']
    Returns:

    '''
    tasks = data.task_id.unique()
    features = []
    for task in tasks:
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        # pad the inputs with zeros along dim=0 to make all inputs the same length as max_input_length
        inputs = np.pad(inputs, ((0, max_input_length-inputs.shape[0]), (0,0)), 'constant', constant_values=np.nan)
        features.append(inputs)
    features = np.stack(features)

    # plot the mean correlation between features over time
    f, ax = plt.subplots(1, 1, figsize=(7,7))
    for which_feature in range(num_features):
        corr = np.array([np.corrcoef(features[:, ii, which_feature].flatten(), features[:, ii+time_shift, which_feature].flatten())[0,1] for ii in range(features.shape[1]-time_shift)]) 
        ax.plot(corr, label= 'corr($x_{}(t)$, $x_{}(t+{})$)'.format(which_feature+1,which_feature+1, time_shift), color=COLORS['feature_{}'.format(which_feature+1)], linewidth=2)
    ax.set_title(f'Temporal correlation w/ time shift of {time_shift}', fontsize=FONTSIZE)
    ax.set_xlabel('Trials', fontsize=FONTSIZE)
    ax.set_ylabel('Correlation coefficient', fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.xticks(fontsize=FONTSIZE-2)
    sns.despine()
    plt.legend(fontsize=FONTSIZE-2, frameon=False)
    plt.show()
    
# plot trial-by-trial performance of baseline models
def plot_trial_by_trial_performance(data, fit_upto_trial, plot_last_trials, num_trials=None):
    
    # keep only trial_id upto num_trials for all tasks
    data = data[data.trial_id<=num_trials] if num_trials is not None else data

    accuracy_lm, accuracy_svm, scores = evaluate_data_against_baselines(data, fit_upto_trial, num_trials)
    accuracy_lm = [acc[-plot_last_trials:, 0] for acc in scores if len(acc)>=plot_last_trials]
    accuracy_svm = [acc[-plot_last_trials:, 1] for acc in scores if len(acc)>=plot_last_trials]
    num_tasks= data.task_id.nunique()
    # accuracy_lm = [acc[-plot_last_trials:] for acc in accuracy_lm if len(acc)>=plot_last_trials]
    # accuracy_svm = [acc[-plot_last_trials:] for acc in accuracy_svm if len(acc)>=plot_last_trials]
    f, ax = plt.subplots(1, 1, figsize=(7,7))   
    num_tasks = len(accuracy_lm)
    ax.plot(np.arange(-plot_last_trials, 0), torch.stack(accuracy_lm).sum(0)/num_tasks, label='Logistic Regression', color=COLORS['lr'])
    ax.plot(np.arange(-plot_last_trials, 0), torch.stack(accuracy_svm).sum(0)/num_tasks, label='SVM', color=COLORS['svm'])
    #ax.set_ylim([0., .8])
    ax.hlines(0.5, -plot_last_trials, 0, color='k', linestyles='dotted', lw=4)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.xticks(fontsize=FONTSIZE-2)
    ax.set_xlabel('Trial', fontsize=FONTSIZE)
    ax.set_ylabel('Mean accuracy (over tasks)', fontsize=FONTSIZE)
    ax.set_title(f'Performance over trials', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE-2, loc='lower right')
    sns.despine()
    f.tight_layout()
    plt.show()

# plot histogram of binned data
def plot_histogram_binned_data(data, num_bins, min_value=0, max_value=1):

    bin_counts, target_counts = bin_data_points(num_bins, data, min_value, max_value)    
    b_counts = np.stack(bin_counts)-np.stack(target_counts) 
    a_counts = np.stack(target_counts)  

    f, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.hist([a_counts, b_counts], label=['A', 'B'], color=[COLORS['a'], COLORS['b']])
    plt.legend(fontsize=FONTSIZE-2,  loc="upper center", bbox_to_anchor=(.45, 1.1), ncol=3, frameon=False)
    ax.set_ylabel('Bin counts', fontsize=FONTSIZE)
    ax.set_xlabel('Number of points per unit volume', fontsize=FONTSIZE) #$a_{name_trials}$
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

# plot histogram of binned data
def plot_sorted_volumes(data, num_bins, min_value=0, max_value=1):

    bin_counts, target_counts = bin_data_points(num_bins, data, min_value, max_value)    
    b_counts = np.stack(bin_counts)-np.stack(target_counts) 
    a_counts = np.stack(target_counts)  
    b_counts.sort()
    a_counts.sort()

    f, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(np.arange(len(b_counts)), b_counts[::-1], alpha=0.9, label='B',  color=COLORS['b'], lw=3)
    ax.plot(np.arange(len(a_counts)), a_counts[::-1], alpha=0.9, label='A',  color=COLORS['a'], lw=3)

    ax.set_ylabel('Number of points per unit volumes', fontsize=FONTSIZE)
    ax.set_xlabel('Sorted volumes (in descending order)', fontsize=FONTSIZE) #$a_{name_trials}$
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()


def plot_data_stats(data):

    all_corr, all_coef, posterior_logprob, _ = return_data_stats(data)
    
    fig, axs = plt.subplots(1, 3,  figsize=(14,6))
    sns.histplot(np.array(all_corr), ax=axs[0], bins=10, stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(np.array(all_coef), ax=axs[1], bins=11, binrange=(-10, 10), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(posterior_logprob[:, 0].exp().detach(), ax=axs[2], bins=5, stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    
    #axs[2].set_ylim(0, 0.5)
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(0, 0.25)
    axs[1].set_ylim(0, 0.3)
    
    axs[0].set_yticks(np.arange(0, 0.25, 0.05))
    axs[1].set_yticks(np.arange(0, 0.35, 0.05))
    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)


    axs[0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')

    axs[0].set_xlabel('Input correlation', fontsize=FONTSIZE)
    axs[1].set_xlabel('Regression coefficients', fontsize=FONTSIZE)
    axs[2].set_xlabel('Linearity', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.show()


def plot_cue_validity(data):

    _, _, _, feature_coef = return_data_stats(data)

    # plot histogram of fitted regression coefficients for each feature from feature_coef (np.arrray: num_tasks x num_features) 
    # use COLORS['feature_1'] for  feature_1, COLORS['feature_2'] for feature_2, etc.
    f, ax = plt.subplots(1, 1, figsize=(7,7))
    for i in range(feature_coef.shape[1]):
        sns.histplot(feature_coef[:, i], ax=ax, bins=10, binrange=(-10, 10), stat='probability', edgecolor='w', linewidth=1, color=COLORS[f'feature_{i+1}'])
    ax.set_xlabel('Regression coefficients', fontsize=FONTSIZE)
    ax.set_ylabel('Percentage', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

    return feature_coef

def plot_probability_same_class_versus_distance(data):

    distances, probabilities = probability_same_target_vs_distance(data, random=False)
    dists, probs = np.nanmean(distances,0), np.nanmean(probabilities, 0)
    # plot probability vs distance
    f, ax = plt.subplots(1, 1, figsize=(7,7))
    sns.regplot(probs, dists, ax=ax, ci=95, color=COLORS['stats'], \
                scatter_kws={'s': 100, 'alpha': 0.5}, line_kws={'lw': 4, 'color': '#007977'}, \
                    truncate=True)

    #ax.set_title(f'Difference between p(target=A) vs distance between datapoints')
    ax.set_xlabel('L2 distance between datapoints', fontsize=FONTSIZE)
    ax.set_ylabel('|$p_{1}$(A)-$p_{2}$(A)|', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

def plot_per_task_features(df):

    df['input_stripped'] = df['input'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
    df[['feature1', 'feature2', 'feature3']] = pd.DataFrame(df['input_stripped'].to_list(), index=df.index)
    df = df.groupby('task_id').agg({'feature1':list, 'feature2':list, 'feature3':list, 'target':list}).reset_index()
    sample_tasks = np.random.choice(df.task_id.unique(), 5, replace=False)

    # plot the histogram of values taken by the three features in the dataset
    # each feature as separate subplot for 5 tasks
    f, axs = plt.subplots(5, 3, figsize=(10,10))
    for i, task in enumerate(sample_tasks):
        sns.histplot(df[df.task_id==task].feature1.values[0], ax=axs[i, 0], bins=10, stat='probability', edgecolor='w', linewidth=1, color=COLORS['feature_1'])
        sns.histplot(df[df.task_id==task].feature2.values[0], ax=axs[i, 1], bins=10, stat='probability', edgecolor='w', linewidth=1, color=COLORS['feature_2'])
        sns.histplot(df[df.task_id==task].feature3.values[0], ax=axs[i, 2], bins=10, stat='probability', edgecolor='w', linewidth=1, color=COLORS['feature_3'])
        axs[i, 0].set_xlabel('')
        axs[i, 1].set_xlabel('')
        axs[i, 2].set_xlabel('')
        axs[i, 0].set_ylabel('')
        axs[i, 1].set_ylabel('')
        axs[i, 2].set_ylabel('')
        axs[i, 0].set_title(f'Task {task}')
    axs[0, 0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[2, 0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[4, 0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[4, 0].set_xlabel('Feature 1', fontsize=FONTSIZE)
    axs[4, 1].set_xlabel('Feature 2', fontsize=FONTSIZE)
    axs[4, 2].set_xlabel('Feature 3', fontsize=FONTSIZE)
    sns.despine()
    f.tight_layout()
    plt.show()

    
def errors_metalearner(env_name, model_path, mode='test', shuffle_trials=False, num_trials=96, num_runs=5):

    _, model_choices, true_choices, sequences = evaluate_1d(env_name=env_name, \
                model_path=model_path, \
                mode=mode, shuffle_trials=shuffle_trials, \
                return_all=True)
    
    cum_sum = np.array(sequences).cumsum()
    errors = np.zeros((4, len(cum_sum)))
    model_choices = model_choices.round()
    for task_idx, seq in enumerate(cum_sum[:-1]):
        
        # use model_choices and true_choies to categorise error types
        true_positives = np.where((model_choices[seq:seq+np.diff(cum_sum)[task_idx]]==1) & (true_choices[seq:seq+np.diff(cum_sum)[task_idx]]==1))[0]
        true_negatives = np.where((model_choices[seq:seq+np.diff(cum_sum)[task_idx]]==0) & (true_choices[seq:seq+np.diff(cum_sum)[task_idx]]==0))[0]
        false_positives = np.where((model_choices[seq:seq+np.diff(cum_sum)[task_idx]]==1) & (true_choices[seq:seq+np.diff(cum_sum)[task_idx]]==0))[0]
        false_negatives = np.where((model_choices[seq:seq+np.diff(cum_sum)[task_idx]]==0) & (true_choices[seq:seq+np.diff(cum_sum)[task_idx]]==1))[0]

        # save the four error types
        len_task = cum_sum[task_idx+1]-cum_sum[task_idx]
        errors[0,task_idx] = len(true_positives)/len_task
        errors[1,task_idx] = len(true_negatives)/len_task
        errors[2,task_idx] = len(false_positives)/len_task
        errors[3,task_idx] = len(false_negatives)/len_task

    # plot a bar plot of the mean number of error types across tasks
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.bar(np.arange(4), errors.mean(1), color=COLORS['stats'])
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['True positives', 'True negatives', 'False positives', 'False negatives'], rotation=45, fontsize=FONTSIZE-2)
    ax.set_ylabel('Mean number of errors', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()   
    f.tight_layout()
    plt.show()
