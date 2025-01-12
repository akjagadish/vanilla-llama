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
from evaluate import evaluate_1d, evaluate_metalearner
from utils import evaluate_data_against_baselines, bin_data_points
from utils import probability_same_target_vs_distance
from envs import NosofskysTask
import json
from collections import Counter

# set plotting parameters
COLORS = {'a':'#117733', 
          'b':'#96CAA7',
          'lr':'#88CCEE',#882255
          'svm':'#CC6677',
          'optimal': '#D6BF4D',
          'feature_1':'#882255', #332288
          'people':'#882255', #AA4499',
          'stats':'#44AA99', 
          'feature_2':'#44AA99',
          'simple_grammar_constrained_noncompositonal':'#EF9EBB',
          'feature_3':'#E2C294', #'#0571D0', 
          'metal':'#DA9138', #"#D55E00", 
          'people2': '#748995',
          'metal2': '#173b4f'
          }
FONTSIZE=20

SYS_PATH = '/u/ajagadish/vanilla-llama' #f'/raven/u/ajagadish/vanilla-llama

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
    num_targets = np.stack([(data[data.task_id==task_id].target==data[data.task_id==task_id].target.unique()[0]).sum() for task_id in data.task_id.unique()])
    expected_number_points = np.nanmean(np.array([data[data.task_id==ii].trial_id.max()+1 for ii in np.arange(num_tasks)]))
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    bar_positions = [0, 0.55]
    colors = ['#173b4f', '#8b9da7'] #'#748995', '#5d7684', '#456272', '#2e4f61',
    ax.bar(bar_positions, [num_targets.mean(), expected_number_points-num_targets.mean()], color=colors, width=0.4)
    ax.errorbar(bar_positions, [num_targets.mean(), expected_number_points-num_targets.mean()], \
                yerr=[num_targets.std()/np.sqrt(len(num_targets)-1), (expected_number_points-num_targets).std()/np.sqrt(len(num_targets)-1)], \
                c='k', lw=3, fmt="o")
    ax.set_ylabel('# points per class per task', fontsize=FONTSIZE)
    ax.set_xlabel('Category', fontsize=FONTSIZE) 
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(categories, fontsize=FONTSIZE-2)  # Assign category names to x-tick labels
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

    # save figure
    f.savefig(f'{SYS_PATH}/categorisation/figures/label_balance.svg', bbox_inches='tight', dpi=300)

# plot mean number of tasks
def plot_mean_number_tasks(data):
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    expected_number_points = np.array([data[data.task_id==ii].trial_id.max()+1 for ii in np.unique(data.task_id)])
    sns.histplot(expected_number_points, kde=False, bins=50, color='#8b9da7')
    plt.axvline(expected_number_points.mean(), color=COLORS['metal2'], linestyle='--', label=f'Mean: {expected_number_points.mean():.2f}', linewidth=2)
    plt.legend(fontsize=FONTSIZE-4,  loc="lower right", frameon=False)
    ax.set_ylabel('Counts', fontsize=FONTSIZE)
    ax.set_xlabel('Number of data points per task', fontsize=FONTSIZE) #$a_{name_trials}$
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

    # save figure
    f.savefig(f'{SYS_PATH}/categorisation/figures/mean_number_tasks.svg', bbox_inches='tight', dpi=300)

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
def plot_trial_by_trial_performance(data, fit_upto_trial, plot_last_trials, num_trials=None, backwards=False):
    
    num_tasks = data.task_id.nunique()
    accuracy_lm, accuracy_svm, scores = evaluate_data_against_baselines(data, fit_upto_trial, num_trials)
    # if backwards: 
    #     accuracy_lm2 = [acc[-plot_last_trials:, 0] for acc in scores if len(acc)>=plot_last_trials]
    #     accuracy_svm = [acc[-plot_last_trials:, 1] for acc in scores if len(acc)>=plot_last_trials]
    # else:
    #     accuracy_lm2 = [acc[:plot_last_trials, 0] for acc in scores if len(acc)>=plot_last_trials]
    #     accuracy_svm = [acc[:plot_last_trials, 1] for acc in scores if len(acc)>=plot_last_trials]
    # accuracy_lm = [acc[-plot_last_trials:] for acc in accuracy_lm if len(acc)>=plot_last_trials]
    # accuracy_svm = [acc[-plot_last_trials:] for acc in accuracy_svm if len(acc)>=plot_last_trials]

    f, ax = plt.subplots(1, 1, figsize=(5,5))   
    num_tasks = len(accuracy_lm)
    x_labels = np.arange(-plot_last_trials, 0) if backwards else np.arange(0, plot_last_trials)
    # plot lm curve
    mean_lm = torch.stack(accuracy_lm).sum(0)/num_tasks
    std_lm = torch.stack(accuracy_lm).float().std(0)/np.sqrt(num_tasks-1)
    COLORS['svm'] = '#173b4f'
    COLORS['lr'] = '#8b9da7'
    ax.plot(x_labels, mean_lm, label='Logistic Regression', color=COLORS['lr'])
    ax.fill_between(x_labels, mean_lm-std_lm, mean_lm+std_lm, color=COLORS['lr'], alpha=0.2)
    # plot svm curve
    mean_svm = torch.stack(accuracy_svm).sum(0)/num_tasks
    std_svm = torch.stack(accuracy_svm).float().std(0)/np.sqrt(num_tasks-1)
    ax.plot(x_labels, mean_svm, label='SVM', color=COLORS['svm'])
    ax.fill_between(x_labels, mean_svm-std_svm, mean_svm+std_svm, color=COLORS['svm'], alpha=0.2)
    #ax.set_ylim([0., .8])
    if backwards: 
        ax.hlines(0.5, -plot_last_trials, 0, color='k', linestyles='dotted', lw=4)
    else:
        ax.hlines(0.5, 0, plot_last_trials, color='k', linestyles='dotted', lw=4)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.xticks(fontsize=FONTSIZE-2)
    ax.set_xlabel('Trial', fontsize=FONTSIZE)
    ax.set_ylabel('Mean accuracy (over tasks)', fontsize=FONTSIZE)
    #ax.set_title(f'Performance over trials', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE-2, loc='lower right', frameon=False)
    sns.despine()
    f.tight_layout()
    plt.show()

    # save figure
    f.savefig(f'{SYS_PATH}/categorisation/figures/trial_by_trial_performance.svg', bbox_inches='tight', dpi=300)

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

def plot_data_stats(data, poly_degree=2):

    all_corr, all_coef, posterior_logprob, per_feature_coef, per_feature_corrs, gini_coeff, advantage = return_data_stats(data, poly_degree)

    COLORS['stats'] = '#173b4f'
    fig, axs = plt.subplots(1, 3,  figsize=(15,5))
    sns.histplot(np.array(all_corr), ax=axs[0], bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
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

    # save figure
    fig.savefig(f'{SYS_PATH}/categorisation/figures/data_stats.svg', bbox_inches='tight', dpi=300)

    # plot the 3 pairwise correlation between features in separate subplots
    f, axs = plt.subplots(1, 3, figsize=(15,5))
    for i in range(per_feature_corrs.shape[1]):
        sns.histplot(per_feature_corrs[:, i], ax=axs[i], bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color=COLORS[f'feature_{i+1}'])
    axs[0].set_xlabel('feature 1 and feature 2', fontsize=FONTSIZE)
    axs[1].set_xlabel('feature 1 and feature 3', fontsize=FONTSIZE)
    axs[2].set_xlabel('feature 2 and feature 3', fontsize=FONTSIZE)
    axs[0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[0].set_ylim(0, 0.5)
    axs[1].set_ylim(0, 0.5)
    axs[2].set_ylim(0, 0.5)
    axs[0].set_yticks(np.arange(0, 0.5, 0.1))
    axs[1].set_yticks(np.arange(0, 0.5, 0.1))
    axs[2].set_yticks(np.arange(0, 0.5, 0.1))
    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    sns.despine()
    plt.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/correlation_features.svg', bbox_inches='tight', dpi=300)

    # plot the regression coefficients for the 3 features in separate subplots
    f, axs = plt.subplots(1, 3, figsize=(15,5))
    for i in range(per_feature_coef.shape[1]):
        sns.histplot(per_feature_coef[:, i], ax=axs[i], bins=11, binrange=(-10, 10), stat='probability', edgecolor='w', linewidth=1, color=COLORS[f'feature_{i+1}'])
    axs[0].set_xlabel('Coefficient for feature 1', fontsize=FONTSIZE)
    axs[1].set_xlabel('Coefficient for feature 2', fontsize=FONTSIZE)
    axs[2].set_xlabel('Coefficient for feature 3', fontsize=FONTSIZE)
    axs[0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[0].set_ylim(0, 0.5)
    axs[1].set_ylim(0, 0.5)
    axs[2].set_ylim(0, 0.5)
    axs[0].set_yticks(np.arange(0, 0.5, 0.1))
    axs[1].set_yticks(np.arange(0, 0.5, 0.1))
    axs[2].set_yticks(np.arange(0, 0.5, 0.1))
    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    sns.despine()
    plt.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/coefficient_features.svg', bbox_inches='tight', dpi=300)


    # plot the gini coefficients
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    gini_coeff = np.array(gini_coeff)
    gini_coeff = gini_coeff[~np.isnan(gini_coeff)]
    bin_max = np.max(gini_coeff)
    sns.histplot(gini_coeff, ax=ax, bins=11, binrange=(0, bin_max), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    ax.set_xlabel('Gini coefficient', fontsize=FONTSIZE)
    ax.set_ylabel('Percentage', fontsize=FONTSIZE)
    ax.set_ylim(0, 0.3)
    ax.set_yticks(np.arange(0, 0.3, 0.1))
    # set tick size
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    sns.despine()
    plt.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/gini_coefficient.png', bbox_inches='tight', dpi=300)

    
    COLORS['stats'] = '#173b4f'
    posterior_logprob = posterior_logprob[:, 0].exp().detach().numpy().round(4)
    fig, axs = plt.subplots(1, 3,  figsize=(15,5))
    sns.histplot(np.array(all_corr), ax=axs[0], bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(gini_coeff, ax=axs[1], bins=11, binrange=(0, bin_max), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(posterior_logprob, ax=axs[2], bins=5, stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    
    #axs[2].set_ylim(0, 0.5)
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(0, 0.25)
    axs[1].set_ylim(0, 0.3)
    
    axs[0].set_yticks(np.arange(0, 0.25, 0.05))
    axs[1].set_yticks(np.arange(0, 0.3, 0.1))
    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)


    axs[0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')

    axs[0].set_xlabel('Input correlation', fontsize=FONTSIZE)
    axs[1].set_xlabel('Gini coefficient', fontsize=FONTSIZE)
    axs[2].set_xlabel('Linearity', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.show()

    # save figure
    fig.savefig(f'{SYS_PATH}/categorisation/figures/corr_gini_linearity.svg', bbox_inches='tight', dpi=300)



    advantage = np.stack(advantage)
    advantage[advantage<0] = 0  # set values below 0 to 0
    #advantage = 1-(np.stack(advantage)/np.max(np.stack(advantage)))


    COLORS['stats'] = '#173b4f'
    fig, axs = plt.subplots(1, 3,  figsize=(15,5))
    sns.histplot(np.array(all_corr), ax=axs[0], bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(gini_coeff, ax=axs[1], bins=11, binrange=(0, bin_max), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(advantage, ax=axs[2], bins=11, binrange=(-0.05, 1.05), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    #axs[2].set_ylim(0, 0.5)
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(0, 0.25)
    #axs[2].set_xlim(0., 0.5)
    axs[1].set_ylim(0, 0.3)
    
    axs[0].set_yticks(np.arange(0, 0.25, 0.05))
    axs[1].set_yticks(np.arange(0, 0.3, 0.1))
    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)


    axs[0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')

    axs[0].set_xlabel('Input correlation', fontsize=FONTSIZE)
    axs[1].set_xlabel('Gini coefficient', fontsize=FONTSIZE)
    axs[2].set_xlabel('Linearity', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.show()

    # save figure
    fig.savefig(f'{SYS_PATH}/categorisation/figures/corr_gini_adv.svg', bbox_inches='tight', dpi=300)


def plot_data_stats_synthetic(data, poly_degree=2, synthetic_type='linear', dim=None):

    all_corr, all_coef, posterior_logprob, per_feature_coef, per_feature_corrs, gini_coeff, advantage = return_data_stats(data, poly_degree)
    
    gini_coeff = np.array(gini_coeff)
    gini_coeff = gini_coeff[~np.isnan(gini_coeff)]
    bin_max = np.max(gini_coeff)

    posterior_logprob = posterior_logprob[:, 0].exp().detach().numpy().round(4)
    
    advantage = np.stack(advantage)
    advantage[advantage<0] = 0  # set values below 0 to 0
    # advantage = 1-(np.stack(advantage)/np.max(np.stack(advantage)))

    COLORS['stats'] = '#173b4f'
    fig, axs = plt.subplots(1, 3,  figsize=(15,5))
    sns.histplot(np.array(all_corr), ax=axs[0], bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(gini_coeff, ax=axs[1], bins=11, binrange=(0, bin_max), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(posterior_logprob, ax=axs[2], bins=5, stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    #axs[2].set_ylim(0, 0.5)
    axs[0].set_xlim(-1, 1)
    assert dim==3, 'Only works for dim=3'
    axs[1].set_xlim(0., 0.66 if dim==3 else None)   
    axs[2].set_xlim(0., 1.05)
    axs[0].set_ylim(0, 0.25)
    # axs[1].set_ylim(0, 0.3)
    
    axs[0].set_yticks(np.arange(0, 0.25, 0.05))
    axs[1].set_yticks(np.arange(0, 0.3, 0.1))
    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)


    axs[0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')

    axs[0].set_xlabel('Input correlation', fontsize=FONTSIZE)
    axs[1].set_xlabel('Gini coefficient', fontsize=FONTSIZE)
    axs[2].set_xlabel('Linearity', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.show()

    # save figure
    fig.savefig(f'{SYS_PATH}/categorisation/figures/corr_gini_linearity_synthetic{synthetic_type}.svg', bbox_inches='tight', dpi=300)


    COLORS['stats'] = '#173b4f'
    fig, axs = plt.subplots(1, 3,  figsize=(15,5))
    sns.histplot(np.array(all_corr), ax=axs[0], bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(gini_coeff, ax=axs[1], bins=11, binrange=(0, bin_max), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    sns.histplot(advantage, ax=axs[2], bins=11, binrange=(-0.05, 1.05), stat='probability', edgecolor='w', linewidth=1, color=COLORS['stats'])
    #axs[2].set_ylim(0, 0.5)
    axs[0].set_xlim(-1, 1)

    axs[1].set_xlim(0., 0.66 if dim==3 else None)   
    # axs[2].set_xlim(0., 0.5)
    axs[0].set_ylim(0, 0.25)
    # axs[1].set_ylim(0, 0.3)
    
    axs[0].set_yticks(np.arange(0, 0.25, 0.05))
    axs[1].set_yticks(np.arange(0, 0.3, 0.1))
    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)


    axs[0].set_ylabel('Percentage', fontsize=FONTSIZE)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')

    axs[0].set_xlabel('Input correlation', fontsize=FONTSIZE)
    axs[1].set_xlabel('Gini coefficient', fontsize=FONTSIZE)
    axs[2].set_xlabel('Linearity', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.show()

    # save figure
    fig.savefig(f'{SYS_PATH}/categorisation/figures/corr_gini_adv_synthetic{synthetic_type}.svg', bbox_inches='tight', dpi=300)

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

def plot_per_task_features_for_selected_tasks(df, sample_task_per_feature=None):

    df['input_stripped'] = df['input'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
    df[['feature1', 'feature2', 'feature3']] = pd.DataFrame(df['input_stripped'].to_list(), index=df.index)
    df = df.groupby('task_id').agg({'feature1':list, 'feature2':list, 'feature3':list, 'target':list}).reset_index()
    task1, task2, task3 = sample_task_per_feature
    f, axs = plt.subplots(1, 3, figsize=(15,5))

    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    sns.histplot(df[df.task_id==task1].feature1.values[0], ax=axs[0], bins=10, stat='probability', edgecolor='w', linewidth=1, color=COLORS['feature_1'])
    sns.histplot(df[df.task_id==task2].feature2.values[0], ax=axs[1], bins=10, stat='probability', edgecolor='w', linewidth=1, color=COLORS['feature_2'])
    sns.histplot(df[df.task_id==task3].feature3.values[0], ax=axs[2], bins=10, stat='probability', edgecolor='w', linewidth=1, color=COLORS['feature_3'])
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('')
    axs[0].set_ylabel('')
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[0].set_ylim(0., 0.2)
    axs[1].set_ylim(0., 0.2)
    axs[2].set_ylim(0., 0.2)
    axs[0].set_ylabel('probability', fontsize=FONTSIZE-2)
    axs[0].set_xlabel('Shape', fontsize=FONTSIZE-2)
    axs[1].set_xlabel('Size', fontsize=FONTSIZE-2)
    axs[2].set_xlabel('Color', fontsize=FONTSIZE-2)
        
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
            
    sns.despine()
    f.tight_layout()
    plt.show()
    
     # save figure
    f.savefig(f'{SYS_PATH}/categorisation/figures/selected_task_features.svg', bbox_inches='tight', dpi=300)
   
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

def compare_metalearners(env_name=None, model_env=None, experiment='categorisation', tasks=[None], noises=[0.05, 0.1, 0.0], shuffles=[True, False], shuffle_evals=[True, False], num_runs=5, num_trials=96, num_eval_tasks=1113, synthetic=False):

    corrects = np.ones((len(tasks), len(noises), len(shuffles), len(shuffle_evals), num_eval_tasks, num_trials))
    for t_idx, task in enumerate(tasks):
        for n_idx, noise in enumerate(noises):
            for s_idx, shuffle in enumerate(shuffles):
                for se_idx, shuffle_eval in enumerate(shuffle_evals):

                    if experiment=='categorisation':
                        env_path = f'{SYS_PATH}/categorisation/data/{env_name}.csv'
                    else:
                        raise NotImplementedError

                    if synthetic:
                        model_name = f"env={env_name}_{model_env}_noise{noise}_shuffle{shuffle}_run=0_synthetic.pt"
                    else:
                        
                        model_name = f"env={env_name}_{model_env}_noise{noise}_shuffle{shuffle}_run=0.pt"

                    model_path=f"/raven/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}"

                    corrects[t_idx, n_idx, s_idx, se_idx] = evaluate_metalearner(env_path, model_path, experiment, shuffle_trials=shuffle_eval, num_runs=num_runs)
        
    # compuate error rates across trials using corrects
    errors = 1. - corrects.mean(3)

    # compare the error rate over trials between different noise levels meaned over shuffles and shuffle_evals
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    for n_idx, noise in enumerate(noises):
        ax.plot(np.arange(num_trials), errors[:, n_idx].mean(0).mean(0).mean(0), label=f'Noise={noise}', lw=3)
    ax.set_xlabel('Trial', fontsize=FONTSIZE)
    ax.set_ylabel('Error rate', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(fontsize=FONTSIZE-2, frameon=False)
    sns.despine()
    f.tight_layout()
    plt.show()

    # compare the error rate over trials between different shuffle settings meaned over noise levels and shuffle_evals
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    for s_idx, shuffle in enumerate(shuffles):
        ax.plot(np.arange(num_trials), errors[:, :, s_idx].mean(0).mean(0).mean(0), label=f'Shuffle={shuffle}', lw=3)
    ax.set_xlabel('Trial', fontsize=FONTSIZE)
    ax.set_ylabel('Error rate', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(fontsize=FONTSIZE-2, frameon=False)
    sns.despine()
    f.tight_layout()
    plt.show()

    # compare the error rate over trials between different shuffle_eval settings meaned over noise levels and shuffles
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    for se_idx, shuffle_eval in enumerate(shuffle_evals):
        ax.plot(np.arange(num_trials), errors[:, :, :, se_idx].mean(0).mean(0).mean(0), label=f'Shuffle_eval={shuffle_eval}', lw=3)
    ax.set_xlabel('Trial', fontsize=FONTSIZE)
    ax.set_ylabel('Error rate', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(fontsize=FONTSIZE-2, frameon=False)
    sns.despine()
    f.tight_layout()
    plt.show()

def evaluate_nosofsky1994(env_name=None, experiment=None, tasks=[None], beta=1., noises=[0.05, 0.1, 0.0], shuffles=[True, False], shuffle_evals=[True, False], num_runs=5, num_trials=96, num_blocks=1, num_eval_tasks=1113, synthetic=False, rmc=False, nonlinear=False, run=0):

    corrects = np.ones((len(tasks), len(noises), len(shuffles), len(shuffle_evals), num_eval_tasks, num_trials))
    for t_idx, task in enumerate(tasks):
        for n_idx, noise in enumerate(noises):
            for s_idx, shuffle in enumerate(shuffles):
                for se_idx, shuffle_eval in enumerate(shuffle_evals):
                    if synthetic:
                        model_name = f"env={env_name}_noise{noise}_shuffle{shuffle}_run={run}_synthetic{'nonlinear' if nonlinear else ''}.pt"
                    elif rmc:
                        model_name = f"env={env_name}_noise{noise}_shuffle{shuffle}_run={run}_rmc.pt"
                    else:
                        model_name = f"env={env_name}_noise{noise}_shuffle{shuffle}_run={run}.pt"
                    model_path = f"/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}"
                    corrects[t_idx, n_idx, s_idx, se_idx] = evaluate_metalearner(task, model_path, 'shepard_categorisation', \
                                                                                 beta=beta, shuffle_trials=shuffle_eval, num_trials=num_trials, num_runs=num_runs)
        
    # compuate error rates across trials using corrects
    errors = 1. - corrects.mean(3)
    errors = np.mean(errors, axis=(1,2,3))

    # compare the error rate over trials between different tasks meaned over noise levels, shuffles and shuffle_evals
    f, ax = plt.subplots(1, 1, figsize=(6,5))
    colors_mpi_blues = ['#8b9da7', '#748995', '#5d7684', '#456272', '#2e4f61', '#173b4f']
    colors_poster = ['#819BAF', '#A2C0A9', '#E3E2C3', '#E3C495', '#D499AB', '#7C7098']
    colors_grays = ['#E0E1DD', '#B6B9B9', '#8C9295', '#616A72','#37434E','#0D1B2A']
    colors = colors_grays
    # markers for the six types of rules in the plot: circle, cross, plus, inverted triangle, asterisk, triangle
    markers = ['o', 'x', '+', '*', 'v', '^']
    for t_idx, task in enumerate(tasks):
        if num_blocks>1:
            block_errors = np.stack(np.split(errors[t_idx], num_blocks)).mean(1)
            ax.plot(np.arange(1, num_blocks+1), block_errors, label=f'Type {task}', lw=3, color=colors[t_idx], marker=markers[t_idx], markersize=8)
        else:
            ax.plot(np.arange(num_trials), errors[t_idx], label=f'Type {task}', lw=3, color=colors[t_idx])#, marker=markers[t_idx], markersize=8)
    ax.set_xticks(np.arange(1, num_blocks+1) if num_blocks>1 else np.arange(1, num_trials+1)) 
    ax.set_xlabel('Trial' if num_blocks==1 else 'Block', fontsize=FONTSIZE)
    ax.set_ylabel('Error rate', fontsize=FONTSIZE)
    ax.set_ylim([0., .5])
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.45, 1.25), ncol=3)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()

    f.savefig(f'{SYS_PATH}/categorisation/figures/nosofsky1994_metalearner_{model_name}.svg', bbox_inches='tight', dpi=300)
    
def evaluate_nosofsky1988(env_name=None, experiment=1, beta=1., noises=[0.05, 0.1, 0.0], shuffles=[True, False], num_runs=5, num_trials=64, num_blocks=3, num_eval_tasks=64, synthetic=False, nonlinear=False, run=0):
    num_trials = num_blocks*num_trials
    tasks = [[4*num_blocks, None, None], [4*num_blocks, 1, 5], [4*num_blocks, 6, 5]] if experiment==1 else [[4*num_blocks, None, None], [4*num_blocks, 5, 3], [4*num_blocks, 5, 5]]
    correct = np.zeros((len(tasks), len(noises), len(shuffles), num_eval_tasks, num_trials))
    model_choices = np.ones((len(tasks), num_runs, len(noises), len(shuffles), num_eval_tasks, num_trials))
    true_choices = np.ones((len(tasks), num_runs, len(noises), len(shuffles), num_eval_tasks, num_trials))
    labels = np.ones((len(tasks), num_runs, len(noises), len(shuffles), num_eval_tasks, num_trials))
    
    for t_idx, task in enumerate(tasks):
        start_trial = 16*num_blocks if task[1] is None else 0 if task[2]==5 else 8*num_blocks # initial values are set to match the number of trials
        for n_idx, noise in enumerate(noises):
            for s_idx, shuffle in enumerate(shuffles):    
                    if synthetic:
                        model_name = f"env={env_name}_noise{noise}_shuffle{shuffle}_run={run}_synthetic{'nonlinear' if nonlinear else ''}.pt"
                    else:
                        model_name = f"env={env_name}_noise{noise}_shuffle{shuffle}_run={run}.pt"
                    model_path = f"/raven/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}"
                    correct[t_idx, n_idx, s_idx,...,start_trial:], model_choices[t_idx, :, n_idx, s_idx,...,start_trial:],\
                        true_choices[t_idx, :, n_idx, s_idx,...,start_trial:], labels[t_idx, :, n_idx, s_idx,...,start_trial:] \
                              = evaluate_metalearner(task, model_path, 'nosofsky_categorisation', shuffle_trials=None, num_runs=num_runs,\
                                                      beta=beta, return_choices=True, num_trials=num_trials)
    # compute mean choice for each label for all task
    category_means = []
    for t_idx, task in enumerate(tasks):
        performances = []
        for label in range(int(labels.max()+1)):
            performances.append(model_choices[t_idx].squeeze()[...,-1][labels[t_idx].squeeze()[...,-1]==label].mean())      
        category_means.append(performances)


    meta_learning_values = [] if experiment==1 else  np.stack(category_means)[:, 5]
    colors = ['#173b4f', '#8b9da7']
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    bar_positions = np.arange(len(meta_learning_values))*0.5
    ax.bar(bar_positions, meta_learning_values, color=colors[0], width=0.4)
    # ax.set_xlabel('Category label', fontsize=FONTSIZE)
    ax.set_ylabel('Mean choice', fontsize=FONTSIZE)
    ax.set_ylim([0.5, 1.])
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['Base', 'E6(3)', 'E6(5)'], fontsize=FONTSIZE-2)
    ax.set_yticks(np.arange(0.5, 1., 0.1))
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    ax.legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.45, 1.2), ncol=3)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/nosofsky1988_metalearner_{model_name}.svg', bbox_inches='tight', dpi=300)    

def evaluate_levering2020(env_name=None, beta=1., noises=[0.05, 0.1, 0.0], shuffles=[True, False], num_runs=5, num_trials=158, num_eval_tasks=64, synthetic=False, nonlinear=False, run=0):
        
    tasks = ['linear', 'nonlinear']
    correct = np.zeros((len(tasks), len(noises), len(shuffles), num_eval_tasks, num_trials))
    model_choices = np.ones((len(tasks), num_runs, len(noises), len(shuffles), num_eval_tasks, num_trials))
    true_choices = np.ones((len(tasks), num_runs, len(noises), len(shuffles), num_eval_tasks, num_trials))
    labels = np.ones((len(tasks), num_runs, len(noises), len(shuffles), num_eval_tasks, num_trials))
    
    for t_idx, task in enumerate(tasks):
        for n_idx, noise in enumerate(noises):
            for s_idx, shuffle in enumerate(shuffles):                    
                    if synthetic:
                        model_name = f"env={env_name}_noise{noise}_shuffle{shuffle}_run={run}_synthetic{'nonlinear' if nonlinear else ''}.pt"
                    else:
                        model_name = f"env={env_name}_noise{noise}_shuffle{shuffle}_run={run}.pt"
                    model_path = f"/raven/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}"
                    correct[t_idx, n_idx, s_idx], model_choices[t_idx, :, n_idx, s_idx],\
                        true_choices[t_idx, :, n_idx, s_idx], labels[t_idx, :, n_idx, s_idx] \
                            = evaluate_metalearner(task, model_path, 'levering_categorisation', shuffle_trials=None,\
                                                    num_runs=num_runs, return_choices=True, beta=beta, num_trials=num_trials)
        
    # plot the mean accuracy over trials for differet tasks
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    colors = ['#173b4f', '#8b9da7']
    task_names = ['Linear', 'Non-linear']
    for t_idx, task in enumerate(tasks):
        ax.plot(np.arange(num_trials), correct[t_idx].mean(0).mean(0).mean(0), label=f'{task_names[t_idx]}', lw=3, color=colors[t_idx])
    ax.set_xlabel('Trial', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.45, 1.2), ncol=3)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/levering2020_metalearner_{model_name}.svg', bbox_inches='tight', dpi=300)

def replot_levering2020():
    # load json file containing the data
    with open('/raven/u/ajagadish/vanilla-llama/categorisation/data/human/levering2020.json') as json_file:
        data = json.load(json_file)

    performance_linear = data['linear']['y']
    performance_nonlinear = data['nonlinear']['y']
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    colors = ['#173b4f', '#8b9da7']
    ax.plot(np.arange(len(performance_linear)), performance_linear, label=f'Linear', lw=3, color=colors[0])
    ax.plot(np.arange(len(performance_linear)), performance_nonlinear, label=f'Non-Linear', lw=3, color=colors[1])
    ax.set_xlabel('Trial', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.45, 1.2), ncol=3)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/levering2020_humans.svg', bbox_inches='tight', dpi=300)

def replot_nosofsky1988():
    nosofs_task = NosofskysTask(task=[4, None, None], batch_size=1)
    inputs, _, targets = nosofs_task.sample_batch()

    # scatter plot
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.scatter(inputs[:, targets[0].squeeze()==0, 2], inputs[:, targets[0].squeeze()==0, 1])
    ax.scatter(inputs[:, targets[0].squeeze()==1, 2], inputs[:, targets[0].squeeze()==1, 1])
    ax.set_xlabel('Saturation', fontsize=FONTSIZE-2)
    ax.set_ylabel('Brightness', fontsize=FONTSIZE-2)
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/nosofsky1988_task.svg', bbox_inches='tight', dpi=300)

    # base-rate effect
    colors = ['#173b4f', '#8b9da7']
    nosofskys_values = [0.8, 0.87, 0.96]
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    bar_positions = np.arange(len(nosofskys_values))*0.5  #[0, 0.55]
    ax.bar(bar_positions, nosofskys_values, color=colors[0], width=0.4)
    ax.set_ylabel('Mean choice', fontsize=FONTSIZE)
    ax.set_ylim([0.5, 1.])
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['Base', 'E6(3)', 'E6(5)'], fontsize=FONTSIZE-2)
    ax.set_yticks(np.arange(0.5, 1., 0.1))
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    ax.legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.45, 1.2), ncol=3)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/nosofsky1988_humans.svg', bbox_inches='tight', dpi=300)

def replot_nosofsky1994():
    # load json file containing the data
    with open('/raven/u/ajagadish/vanilla-llama/categorisation/data/human/nosofsky1994.json') as json_file:
        data = json.load(json_file)

    # plot the error rates for the six types of rules in data
    f, ax = plt.subplots(1, 1, figsize=(6,5))
    colors_mpi_blues = ['#8b9da7', '#748995', '#5d7684', '#456272', '#2e4f61', '#173b4f']
    colors_poster = ['#819BAF', '#A2C0A9', '#E3E2C3', '#E3C495', '#D499AB', '#7C7098']
    colors_cpi = ['#2E3E5F', '#2E4C67', '#2F5A6F', '#2F6877','#2F757E','#30918E']
    colors_purples = ['#432371', '#683F73', '#8C5B75', '#8C5B75','#D59279','#FAAE7B']
    colors_grays = ['#E0E1DD', '#B6B9B9', '#8C9295', '#616A72','#37434E','#0D1B2A']
    colors = colors_grays

    # markers for the six types of rules in the plot: circle, cross, plus, inverted triangle, asterisk, triangle
    markers = ['o', 'x', '+', '*', 'v', '^']
    
    for i, rule in enumerate(data.keys()):
        ax.plot(np.arange(len(data[rule]['y']))+1, data[rule]['y'], label=f'Type {i+1}', lw=3, color=colors[i], marker=markers[i], markersize=8)
    # integer x ticks
    ax.set_xticks(np.arange(len(data[rule]['y']))+1) 
    ax.set_xlabel('Block', fontsize=FONTSIZE)
    ax.set_ylabel('Error rate', fontsize=FONTSIZE)
    # Get the current x-tick locations and labels
    locs, labels = ax.get_xticks(), ax.get_xticklabels()
    # Set new x-tick locations and labels
    ax.set_xticks(locs[::2])
    ax.set_xticklabels(np.arange(len(data[rule]['y']))[::2]+1)
    ax.set_ylim([0., .5])
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.45, 1.25), ncol=3)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/nosofsky1994_humans.svg', bbox_inches='tight', dpi=300)

def longest_consecutive_sequence(arr):
    return np.median(np.diff(np.insert(np.where(np.diff(arr) != 0)[0] + 1, [0, -1], [0, len(arr)])))/len(arr)

def compute_burstiness(signal):
    return (len(signal)-(np.diff(signal)==0).sum())/len(signal)

def plot_burstiness_training_curriculum(data,  num_tasks=10000):

    # sub-select task
    list_tasks = data.task_id.unique()[:num_tasks] 
    data_subselected = data[data['task_id'].isin(list_tasks)]

    burstinesss, shuffled_burstinesss, block_length, shuffled_block_length  = [], [], [], []
    for task_id in list_tasks:
        y = data_subselected[data_subselected.task_id==task_id]['target'].to_numpy()
        signal = np.unique(y, return_inverse=True)[1]+1
        # data_As = data_subselected[data_subselected.task_id==task_id].target.values
        # signal = np.stack([2. if val=='A' else 1. for val in data_As])
        
        burstinesss.append(compute_burstiness(signal))
        block_length.append(longest_consecutive_sequence(signal))
        
        np.random.shuffle(signal)
        shuffled_burstinesss.append(compute_burstiness(signal))
        shuffled_block_length.append(longest_consecutive_sequence(signal))

    categories = ['original', 'shuffled']
    f, ax = plt.subplots(1, 1, figsize=(7,7))
    colors = ['#173b4f', '#8b9da7'] 
    a = plt.hist([np.stack(block_length), np.stack(shuffled_block_length)], bins=10, color=colors, label=categories);
    ax.set_ylabel('#Counts', fontsize=FONTSIZE)
    ax.set_xlabel('Median block length', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(frameon=False, fontsize=FONTSIZE-5)
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/claude_median_block_length.png', bbox_inches='tight')

    # histogram of burstiness
    categories = ['original', 'shuffled']
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    bar_positions = [0, 0.55]
    colors = ['#173b4f', '#8b9da7'] 
    ax.hist([np.stack(burstinesss), np.stack(shuffled_burstinesss)], density=False, color=colors, label=categories)
    ax.set_ylabel('Counts', fontsize=FONTSIZE)
    ax.set_xlabel('Bursitiness', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(frameon=False, fontsize=FONTSIZE-5, loc='upper left')
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/claude_burstiness_histogram.png', bbox_inches='tight')

    # compare variance of burstiness
    categories = ['original', 'shuffled']
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    bar_positions = [0, 0.55]
    colors = ['#173b4f', '#8b9da7']    
    ax.bar(bar_positions, [np.array(burstinesss).mean(), np.array(shuffled_burstinesss).mean()], color=colors, width=0.4)
    ax.errorbar(bar_positions, [np.array(burstinesss).mean(), np.array(shuffled_burstinesss).mean()],\
                yerr=[np.array(burstinesss).std(), np.array(shuffled_burstinesss).std()], \
                c='k', lw=3, fmt="o") #/np.sqrt(num_tasks-1)
    ax.set_xlabel('LLM generated curriculum', fontsize=FONTSIZE)
    ax.set_ylabel('Bursitiness', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(categories, fontsize=FONTSIZE-2)  # Assign category names to x-tick labels
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/claude_burstiness.png', bbox_inches='tight')

def plot_frequency_tasklabels(file_name, path='/u/ajagadish/vanilla-llama/categorisation/data/tasklabels', feature_names=True, pairs=True, top_labels=50):

    df = pd.read_csv(f'{path}/{file_name}.csv')
    df.feature_names = df['feature_names'].apply(lambda x: eval(x))
    df.category_names = df['category_names'].apply(lambda x: eval(x))
    
    def to_lower(ff):
        return [x.lower() for x in ff]
    
    df.feature_names = df['feature_names'].apply(lambda x: to_lower(x))
    df.category_names = df['category_names'].apply(lambda x: to_lower(x))

    # name of the column containing the feature names
    column_name = 'feature_names' if feature_names else 'category_names'
    # count of number of times a type of features occurs
    list_counts = Counter([tuple(features) for features in df[column_name]] if pairs else np.stack(df[column_name].values).reshape(-1))

    # sort the Counter by counts in descending order
    sorted_list_counts = sorted(list_counts.items(), key=lambda x: x[1], reverse=True)

    # extract the counts and names for the top 50 labels
    task_labels = np.array([task_label[0] for task_label in sorted_list_counts[:top_labels]])
    label_counts= np.array([task_label[1] for task_label in sorted_list_counts[:top_labels]])
    label_names = ['-'.join(task_labels[idx]) for idx in range(len(task_labels))] if pairs else task_labels

    # plot the bars of labels and counts
    f, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.bar(label_names, label_counts)
    plt.xticks(label_names, label_names, rotation=90, fontsize=FONTSIZE-6.5)
    plt.yticks(fontsize=FONTSIZE-6)
    ax.set_xlabel('Feature Names' if feature_names else 'Category Names', fontsize=FONTSIZE)
    ax.set_ylabel('Counts', fontsize=FONTSIZE)
    ax.set_title(f'Top {top_labels} Tasks', fontsize=FONTSIZE)
    sns.despine()
    f.tight_layout()
    plt.show()
    
    f.savefig(f'{SYS_PATH}/categorisation/figures/frequency_plot_tasklabels_{column_name}_paired={pairs}_top{top_labels}.png', bbox_inches='tight', dpi=300)

def evaluate_smith1998(env_name=None, experiment=None, tasks=[None], beta=1., noises=[0.05, 0.1, 0.0], shuffles=[True, False], shuffle_evals=[True, False], num_runs=5, num_trials=96, num_blocks=1, num_eval_tasks=1113, synthetic=False, run=0, nonlinear=False):
    tasks = ['linear', 'nonlinear'] if tasks[0] is None else tasks
    corrects = np.ones((len(tasks), len(noises), len(shuffles), len(shuffle_evals), num_eval_tasks, num_trials))
    
    for t_idx, task in enumerate(tasks):
        for n_idx, noise in enumerate(noises):
            for s_idx, shuffle in enumerate(shuffles):
                for se_idx, shuffle_eval in enumerate(shuffle_evals):
                    if synthetic:
                        model_name = f"env={env_name}_noise{noise}_shuffle{shuffle}_run={run}_synthetic{'nonlinear' if nonlinear else ''}.pt"
                    else:
                        model_name = f"env={env_name}_noise{noise}_shuffle{shuffle}_run={run}.pt"
                    model_path = f"{SYS_PATH}/categorisation/trained_models/{model_name}"
                    corrects[t_idx, n_idx, s_idx, se_idx] = evaluate_metalearner(task, model_path, 'smith_categorisation', \
                                                                                 beta=beta, shuffle_trials=shuffle_eval, num_runs=num_runs, num_trials=num_trials)
                    
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    colors = ['#173b4f', '#8b9da7']
    task_names = ['Linear', 'Non-linear']
    for t_idx, task in enumerate(tasks):
        if num_blocks==1:
            ax.plot(np.arange(num_trials), np.mean(corrects[t_idx], axis=(0,1,2,3)), label=f'{task_names[t_idx]}', lw=3, color=colors[t_idx])
        else:
            ax.plot(np.arange(num_blocks), np.stack(np.split(np.mean(corrects[t_idx], axis=(0,1,2,3)), num_blocks)).mean(1), label=f'{task_names[t_idx]}', lw=3, color=colors[t_idx])
    ax.set_xlabel('Trial', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    ax.set_ylim([0.6, 1.])
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.45, 1.2), ncol=3)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()      
    f.savefig(f'{SYS_PATH}/categorisation/figures/smiths_metalearner_{model_name}.svg', bbox_inches='tight', dpi=300)

def compare_categorisation_model_fits_learning(task_name = 'smithstask'):
    
    r2_gcm = np.load(f'../data/meta_learner/r2_gcm_{task_name}.npy')
    r2_pm = np.load(f'../data/meta_learner/r2_pm_{task_name}.npy')
    num_blocks = r2_gcm.shape[-1]
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.plot(range(num_blocks), r2_gcm[:, 0].mean(0), label='gcm')
    ax.fill_between(range(num_blocks), r2_gcm[:, 0].mean(0) - r2_gcm[:, 0].std(0), r2_gcm[:, 0].mean(0) + r2_gcm[:, 0].std(0), alpha=0.3)
    ax.plot(range(num_blocks), r2_pm[:, 0].mean(0), label='prototype')
    ax.fill_between(range(num_blocks), r2_pm[:, 0].mean(0) - r2_pm[:, 0].std(0), r2_pm[:, 0].mean(0) + r2_pm[:, 0].std(0), alpha=0.3)
    ax.set_xlabel('Block', fontsize=FONTSIZE)
    ax.set_ylabel('pseudo-R$^2$', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.45, 1.2), ncol=2)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()      
    f.savefig(f'{SYS_PATH}/categorisation/figures/fit_gcm_pm_learningtrials_{task_name}.svg', bbox_inches='tight', dpi=300)

def model_comparison(list_models=None, task_name = 'smithstask'):

    task_titles = {'smithstask': 'Smith and Minda (1998)', 'levering2020': 'Levering et al. (2020)', \
                  'nosofsky1994': 'Nosofsky et al. (1994)', 'nosofsky1988': 'Nosofsky et al. (1988)',\
                  'badham2017': 'Badham et al. (2017)'}
    
    models = ['badham2017_env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_beta_sweep.npy',\
              'badham2017_env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear_beta_sweep', \
              'badham2017_env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic_beta_sweep.npy']
    models = list_models if list_models is not None else models
    
    nlls,fitted_betas = [], []
    for model_name in models:
        fits =  np.load(f'../model_comparison/{model_name}.npz')
        betas, pnlls, pr2s = fits['betas'], fits['nlls'], fits['pr2s']
        pr2s = np.array(pr2s)
        min_nll_index = np.argmin(np.stack(pnlls), 0)
        pr2s_min_nll = np.stack([pr2s[min_nll_index[idx], idx] for idx in range(pr2s.shape[1])])
        nlls_min_nlls = np.stack([pnlls[min_nll_index[idx], idx] for idx in range(pnlls.shape[1])])
        nlls.append(nlls_min_nlls)
        fitted_betas.append(betas[min_nll_index])

    nlls = np.array(nlls)
    num_participants = len(nlls[0])

    # compare mean nlls across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    bar_positions = np.arange(len(nlls))*0.5
    colors = ['#173b4f', '#8b9da7', '#5d7684']
    ax.bar(bar_positions, nlls.mean(1), color=colors, width=0.4)
    ax.errorbar(bar_positions, nlls.mean(1), yerr=nlls.std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    ax.axhline(y=-np.log(0.5)*384, color='k', linestyle='--', lw=3)
    ax.set_xlabel('Meta-learner trained on', fontsize=FONTSIZE)
    ax.set_ylabel('NLL', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # set x-tick positions to bar_positions
    ax.set_xticklabels(['LLM', 'Linear', 'Non-linear'], fontsize=FONTSIZE-2)  # assign category names to x-tick labels
    ax.set_title(f'Model comparison for {task_titles[task_name]}', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

    # plot histogram of fitted betas for each models in a different subplot
    f, ax = plt.subplots(1, len(models), figsize=(5,15))
    colors = ['#173b4f', '#8b9da7', '#5d7684']
    for m_idx, model in enumerate(models):
        ax[m_idx].hist(fitted_betas[m_idx], color=colors[m_idx])
        ax[m_idx].set_xlabel('Beta', fontsize=FONTSIZE)
        ax[m_idx].set_ylabel('Counts', fontsize=FONTSIZE)
        ax[m_idx].set_title(f'{task_titles[task_name]}', fontsize=FONTSIZE)
        sns.despine()

def model_comparison_badham2017(FIGSIZE=(6,5)):
    models = ['badham2017_env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_soft_sigmoid_differential_evolution',\
              'badham2017_env=rmc_tasks_dim3_data100_tasks11499_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_rmc_soft_sigmoid_differential_evolution',
              'badham2017_env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear_soft_sigmoid_differential_evolution',\
              'badham2017_env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic_soft_sigmoid_differential_evolution',\
              'badham2017_gcm_runs=1_iters=1_blocks=1_loss=nll',\
              'badham2017_rulex_runs=1_iters=1_blocks=1_loss=nll_exception=True',\
              'badham2017_rulex_runs=1_iters=1_blocks=1_loss=nll_exception=False',\
              'badham2017_pm_runs=1_iters=1_blocks=1_loss=nll',\
                ]
    nlls,fitted_betas, r2s = [], [], []
    model_accs = []
    bics = []
    NUM_TASKS = 4
    NUM_TRIALs = 96
    num_trials = NUM_TRIALs*NUM_TASKS
    # FONTSIZE = 16
    # MODELS = ['ERMI', 'RMC-MI', 'L-MI', 'PFN-MI', 'GCM', 'Rulex', 'Rule',  'PM']
    MODELS = ['ERMI', 'RMC-\nMI', 'L-MI', 'PFN-\nMI', 'GCM', 'Rulex', 'Rule',  'PM']


    for model_name in models:
        fits =  np.load(f'{SYS_PATH}/categorisation/data/model_comparison/{model_name}.npz')
        if 'model=transformer' in model_name:
            betas, pnlls, pr2s = fits['betas'], fits['nlls'], fits['pr2s']
            model_accs.append(fits['accs'].max(0).mean())
            nlls_min_nlls = np.array(pnlls).squeeze()
            pr2s_min_nll = np.array(pr2s).squeeze()
            fitted_betas.append(betas)
            num_parameters = 1 
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters
        elif ('gcm' in model_name) or ('pm' in model_name):
            betas, pnlls, pr2s = fits['params'], fits['lls'], fits['r2s']
            # summing the fits for the four conditions separately; hence the total number of parameters is model_parameters*NUM_TASKS
            nlls_min_nlls = np.array(pnlls).squeeze().sum(1)
            pr2s_min_nll = np.array(pr2s).squeeze().mean(1)
            num_parameters = 5*NUM_TASKS if ('gcm' in model_name) else 11*NUM_TASKS
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters
            fitted_betas.append(betas.squeeze()[..., 1].mean(1))
        elif ('rulex' in model_name):
            betas, pnlls, pr2s = fits['params'], fits['lls'], fits['r2s']
            nlls_min_nlls = np.array(pnlls).squeeze().sum(1)
            pr2s_min_nll = np.array(pr2s).squeeze().mean(1)
            num_parameters = 2*NUM_TASKS
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters

        nlls.append(nlls_min_nlls)
        r2s.append(pr2s_min_nll)
        bics.append(bic)

    # keep models and choose colors
    num_participants = len(nlls[0])
    MODELS = MODELS[:len(nlls)]
    # set colors depending on number of models in MODELS
    colors = ['#173b4f', '#8b9da7', '#5d7684', '#2f4a5a', '#0d2c3d', '#4d6a75', '#748995', '#a2c0a9'][:len(nlls)]


    # compare mean nlls across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(nlls))*0.5
    ax.bar(bar_positions, np.array(nlls).mean(1), color=colors, width=0.4)
    ax.errorbar(bar_positions, np.array(nlls).mean(1), yerr=np.array(nlls).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    # add chance level line for 616 trials with binary choices
    ax.axhline(y=-np.log(0.5)*num_trials, color='k', linestyle='--', lw=3)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('NLL', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-5)  # Assign category names to x-tick labels
    # ax.set_title(f'Model comparison for Badham et al. (2017)', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

    # compare mean BICS across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(bics))*1.5
    ax.bar(bar_positions, np.array(bics).mean(1), color=colors, width=1.)
    ax.errorbar(bar_positions, np.array(bics).mean(1), yerr=np.array(bics).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    # add chance level line for 616 trials with binary choices
    # ax.axhline(y=-np.log(0.5)*num_trials*2, color='k', linestyle='--', lw=3)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('BIC', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-6)  # Assign category names to x-tick labels
    # ax.set_title(f'Model comparison for Badham et al. (2017)', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/bic_badham2017.svg', bbox_inches='tight', dpi=300)

    # compare mean r2s across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(r2s))*0.5
    ax.bar(bar_positions, np.array(r2s).mean(1), color=colors, width=0.4)
    ax.errorbar(bar_positions, np.array(r2s).mean(1), yerr=np.array(r2s).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('R2', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-5)  # Assign category names to x-tick labels

    # ax.set_title(f'Model comparison for  Badham et al. (2017)', fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()   

def model_comparison_devraj2022(FIGSIZE=(6,5)):
    models = ['devraj2022_env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_soft_sigmoid_differential_evolution', \
              'devraj2022_env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_synthetic_soft_sigmoid_differential_evolution', \
              'devraj2022_env=dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=2_syntheticnonlinear_soft_sigmoid_differential_evolution',\
              'devraj2022_gcm_runs=1_iters=1_blocks=1_loss=nll',\
              'devraj2022_pm_runs=1_iters=1_blocks=1_loss=nll',\
              'devraj2022_rulex_runs=1_iters=1_blocks=1_loss=nll_exception=True', \
              'devraj2022_rulex_runs=1_iters=1_blocks=1_loss=nll_exception=False', 
         ]
    nlls,fitted_betas, r2s = [], [], []
    model_accs = []
    bics = []
    NUM_TASKS = 1
    NUM_TRIALs = 616
    num_trials = NUM_TRIALs*NUM_TASKS
    # FONTSIZE = 16
    MODELS = ['ERMI', 'L-MI', 'PFN-\nMI', 'GCM', 'PM', 'Rulex', 'Rule']

    for model_name in models:
        fits =  np.load(f'{SYS_PATH}/categorisation/data/model_comparison/{model_name}.npz')
        if 'model=transformer' in model_name:
            betas, pnlls, pr2s = fits['betas'], fits['nlls'], fits['pr2s']
            model_accs.append(fits['accs'].max(0).mean())
            nlls_min_nlls = np.array(pnlls).squeeze()
            pr2s_min_nll = np.array(pr2s).squeeze()
            fitted_betas.append(betas)
            num_parameters = 1 
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters
        elif ('gcm' in model_name) or ('pm' in model_name):
            betas, pnlls, pr2s = fits['params'], fits['lls'], fits['r2s']
            # summing the fits for the four conditions separately; hence the total number of parameters is model_parameters*NUM_TASKS
            nlls_min_nlls = np.array(pnlls).squeeze()
            pr2s_min_nll = np.array(pr2s).squeeze()
            num_parameters = 8*NUM_TASKS
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters
            fitted_betas.append(betas.squeeze()[:, 1])
        elif ('rulex' in model_name):
            betas, pnlls, pr2s = fits['params'], fits['lls'], fits['r2s']
            nlls_min_nlls = np.array(pnlls).squeeze()
            pr2s_min_nll = np.array(pr2s).squeeze()
            num_parameters = 2*NUM_TASKS
            bic = np.array(nlls_min_nlls)*2 + np.log(num_trials)*num_parameters

        nlls.append(nlls_min_nlls)
        r2s.append(pr2s_min_nll)
        bics.append(bic)

    # keep models and choose colors
    num_participants = len(nlls[0])
    MODELS = MODELS[:len(nlls)]
    # set colors depending on number of models in MODELS
    colors = ['#173b4f', '#8b9da7', '#5d7684', '#2f4a5a', '#0d2c3d', '#4d6a75', '#748995', '#a2c0a9'][:len(nlls)]


    # compare mean nlls across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(nlls))*0.5
    ax.bar(bar_positions, np.array(nlls).mean(1), color=colors, width=0.4)
    ax.errorbar(bar_positions, np.array(nlls).mean(1), yerr=np.array(nlls).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    # add chance level line for 616 trials with binary choices
    ax.axhline(y=-np.log(0.5)*num_trials, color='k', linestyle='--', lw=3)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('NLL', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-5)  # Assign category names to x-tick labels
    # ax.set_title(f'Model comparison for Badham et al. (2017)', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()

    # compare mean BICS across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(bics))*1.5
    ax.bar(bar_positions, np.array(bics).mean(1), color=colors, width=1.)
    ax.errorbar(bar_positions, np.array(bics).mean(1), yerr=np.array(bics).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    # add chance level line for 616 trials with binary choices
    # ax.axhline(y=-np.log(0.5)*num_trials*2, color='k', linestyle='--', lw=3)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('BIC', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-6)  # Assign category names to x-tick labels
    # ax.set_title(f'Model comparison for Badham et al. (2017)', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/bic_devraj2022.svg', bbox_inches='tight', dpi=300)

    # compare mean r2s across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(r2s))*0.5
    ax.bar(bar_positions, np.array(r2s).mean(1), color=colors, width=0.4)
    ax.errorbar(bar_positions, np.array(r2s).mean(1), yerr=np.array(r2s).std(1)/np.sqrt(num_participants-1), c='k', lw=3, fmt="o")
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('R2', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(MODELS, fontsize=FONTSIZE-5)  # Assign category names to x-tick labels

    # ax.set_title(f'Model comparison for  Badham et al. (2017)', fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE-2)
    sns.despine()
    f.tight_layout()
    plt.show() 

def gcm_pm_fitted_simulations():

    models = ['ermi', 'synthetic', 'syntheticnonlinear']
    f, ax = plt.subplots(1, len(models), figsize=(5*len(models),5))
    colors = ['#173b4f', '#37761e']
    for idx, model in enumerate(models):
        fits_gcm = np.load(f'{SYS_PATH}/categorisation/data/fitted_simulation/devraj2022_gcm_runs=1_iters=10_blocks=11_loss=mse_transfer_model={model}.npz')
        fits_pm = np.load(f'{SYS_PATH}/categorisation/data/fitted_simulation/devraj2022_pm_runs=1_iters=10_blocks=11_loss=mse_transfer_model={model}.npz')

        # load mses
        mses_gcm = fits_gcm['lls']
        mses_pm = fits_pm['lls']
        # mean mses across participants: mses are of shape (n_runs=1, n_participants, n_conditions=1, n_blocks)
        mses_gcm = np.squeeze(mses_gcm)
        mses_pm = np.squeeze(mses_pm)
        # std error of mean across participants
        stds_gcm = np.std(mses_gcm, axis=0)/np.sqrt(len(mses_gcm)-1)
        stds_pm = np.std(mses_pm, axis=0)/np.sqrt(len(mses_pm)-1)

        # plot mean mses across participants for each trial segment for both models
        sns.lineplot(x=np.arange(11)+1, y=np.mean(mses_pm, axis=0), ax=ax[idx], color=colors[0], label='PM')
        sns.lineplot(x=np.arange(11)+1, y=np.mean(mses_gcm, axis=0), ax=ax[idx], color=colors[1], label='GCM')
        # add standard error of mean as error bars
        ax[idx].fill_between(np.arange(11)+1, np.mean(mses_pm, axis=0)-stds_pm, np.mean(mses_pm, axis=0)+stds_pm, alpha=0.2, color=colors[0])
        ax[idx].fill_between(np.arange(11)+1, np.mean(mses_gcm, axis=0)-stds_gcm, np.mean(mses_gcm, axis=0)+stds_gcm, alpha=0.2, color=colors[1])
        ax[idx].set_xlabel('Trial segment', fontsize=FONTSIZE)
        ax[idx].set_ylim([0, 1.])
        ax[idx].set_xticks(np.arange(11)+1)
        # set y ticks font size
        ax[idx].tick_params(axis='y', labelsize=FONTSIZE-2)
        ax[idx].set_xticklabels(np.arange(11)+1,fontsize=FONTSIZE-2)
        if idx==0:
            ax[idx].set_ylabel('Error', fontsize=FONTSIZE)
            # remove bounding box around the legend
            ax[idx].legend(frameon=False, fontsize=FONTSIZE-2)
        else:
            # remove legend
            ax[idx].legend([], frameon=False, fontsize=FONTSIZE-2)
        
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/gcm_pm_fitted_simulations.svg', bbox_inches='tight', dpi=300)

def simulate_shepard1961(models=None, tasks=np.arange(1,7), betas=None, num_runs=5, num_trials=96, num_blocks=1, batch_size=64):

    corrects = np.ones((len(models), len(tasks), batch_size, num_trials))
    assert len(models)==len(betas), "Number of models and betas should be the same"
    for m_idx, (model_name, beta) in enumerate(zip(models, betas)):
        if model_name != 'humans':
            for t_idx, task in enumerate(tasks):
                model_path = f"/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}.pt"
                corrects[m_idx, t_idx] = evaluate_metalearner(task, model_path, 'shepard_categorisation', beta=beta, shuffle_trials=True, num_trials=num_trials, num_runs=num_runs)
            
    # models = ['human', 1, 2]
    # corrects = np.random.rand(3, 6, 64, 256)
    # model_name = 'test'

    # compuate error rates across trials using corrects
    errors = 1. - corrects.mean(2)


    # compare the error rate over trials between different tasks meaned over noise levels, shuffles and shuffle_evals
    f, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    colors = ['#E0E1DD', '#B6B9B9', '#8C9295', '#616A72','#37434E','#0D1B2A']
    # markers for the six types of rules in the plot: circle, cross, plus, inverted triangle, asterisk, triangle
    markers = ['o', 'x', '+', '*', 'v', '^']

    for idx, ax in enumerate(axes):

        if models[idx]=='humans':
            # load json file containing the data
            with open('/u/ajagadish/vanilla-llama/categorisation/data/human/nosofsky1994.json') as json_file:
                data = json.load(json_file)
            for i, rule in enumerate(data.keys()):
                ax.plot(np.arange(len(data[rule]['y']))+1, data[rule]['y'], label=f'Type {i+1}', lw=3, color=colors[i], marker=markers[i], markersize=8)
        else:
            for t_idx, task in enumerate(tasks):
                block_errors = np.stack(np.split(errors[idx, t_idx], num_blocks)).mean(1)
                ax.plot(np.arange(1, num_blocks+1), block_errors, label=f'Type {task}', lw=3, color=colors[t_idx], marker=markers[t_idx], markersize=8)

        ax.set_xticks(np.arange(1, num_blocks+1))
        ax.set_xlabel('Block', fontsize=FONTSIZE)
        if idx==0:
            ax.set_ylabel('Error rate', fontsize=FONTSIZE)
        ax.set_ylim([-0.05, .55])
        locs, labels = ax.get_xticks(), ax.get_xticklabels()
        # Set new x-tick locations and labels
        ax.set_xticks(locs[::2])
        ax.set_xticklabels(np.arange(1, num_blocks+1)[::2]+1, fontsize=FONTSIZE-2)
        ax.tick_params(axis='y', labelsize=FONTSIZE-2)

    # add legend that spans across all subplots, in one row, at the center for the subplots, and place it outside the plot 
    f.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=FONTSIZE-2, frameon=False, labels=[f'Type {task}' for task in tasks])
    #axes[int(len(models)/2)].legend(fontsize=FONTSIZE-4, frameon=False,  loc="upper center", bbox_to_anchor=(.5, 1.2), ncol=6)  # place legend outside the plot
    sns.despine()
    f.tight_layout()
    plt.show()
    f.savefig(f'{SYS_PATH}/categorisation/figures/shepard1961_comparison.svg', bbox_inches='tight', dpi=300)