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
    expected_number_points = np.nanmean(np.array([data[data.task_id==ii].trial_id.max()+1 for ii in np.arange(num_tasks)]))
    print(expected_number_points)
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.bar(categories, [num_targets.mean(), expected_number_points-num_targets.mean()], color=[COLORS['a'], COLORS['b']])
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
        ax.plot(corr, label= 'corr($x_{}(t)$, $x_{}(t+{})$)'.format(which_feature+1,which_feature+1, time_shift), color=COLORS['feature_{}'.format(which_feature+1)])
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
    
    accuracy_lm, accuracy_svm = evaluate_data_against_baselines(data, fit_upto_trial, num_trials)
    #print(accuracy_lm, accuracy_svm)
    accuracy_lm = [acc[-plot_last_trials:] for acc in accuracy_lm if len(acc)>=plot_last_trials]
    accuracy_svm = [acc[-plot_last_trials:] for acc in accuracy_svm if len(acc)>=plot_last_trials]
    f, ax = plt.subplots(1, 1, figsize=(7,7))   
    ax.plot(np.arange(-plot_last_trials, 0), torch.stack(accuracy_lm).sum(0)/(data.task_id.max()+1), label='Logistic Regression', color=COLORS['lr'])
    ax.plot(np.arange(-plot_last_trials, 0), torch.stack(accuracy_svm).sum(0)/(data.task_id.max()+1), label='SVM', color=COLORS['svm'])
    #ax.set_ylim([0., .8])
    ax.hlines(0.5, -plot_last_trials, 0, color='k', linestyles='dotted', lw=4)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.xticks(fontsize=FONTSIZE-2)
    ax.set_xlabel('Trial', fontsize=FONTSIZE-2)
    ax.set_ylabel('Mean accuracy (over tasks)', fontsize=FONTSIZE-2)
    ax.set_title(f'Performance over trials', fontsize=FONTSIZE-2)
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
    ax.set_ylabel('Bin counts', fontsize=FONTSIZE-2)
    ax.set_xlabel('Number of points per unit volume', fontsize=FONTSIZE-2) #$a_{name_trials}$
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()


def plot_data_stats(data):

    all_corr, all_coef, posterior_logprob = return_data_stats(data)
    
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