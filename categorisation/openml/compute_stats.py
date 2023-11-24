import openml
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import torch
import sys
sys.path.append("/u/ajagadish/vanilla-llama/categorisation/rl2")
sys.path.append("/u/ajagadish/vanilla-llama/categorisation/")
sys.path.append("/u/ajagadish/vanilla-llama/categorisation/data")
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/data')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/rl2')
from baseline_classifiers import benchmark_baseline_models_regex_parsed_random_points, benchmark_baseline_models_regex_parsed
from baseline_classifiers import LogisticRegressionModel, SVMModel

FONTSIZE=20
color = '#173b4f'

benchmark_suite = openml.study.get_suite('OpenML-CC18') # obtain the benchmark suite
# binary classification tasks with no-nans and less than 100 features
task_ids = [3, 31, 37, 43, 49, 219,\
            3902, 3903, 3913, 3917, 3918,\
            9946, 9952, 9957, 9971, 9978, 10093, \
            10101, 14952, 14965, 146819, 146820, 167120, 167141] 
            # more than 1000 features but no nans: 9910, 9976, 9977, 167125
#TODO: keep only tasks with balanced targets

pooled_correlations, all_bics_linear, all_bics_quadratic, target_balance_ratio = [], [], [], []
accuracy_lm = []
accuracy_svm = []
scores = []
for task_id in task_ids: # iterate over all taskIDs that has passed some prelim checks
    
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    assert (len(task.class_labels) == 2), "Task has more than two classes"
    features, targets = task.get_X_and_y()  # get the data
    assert (not np.isnan(features).any() and (features.shape[1] < 100)), "Task has missing values and greater than 100 features"

    # target balance check
    unique, counts = np.unique(targets, return_counts=True)
    target_balance_ratio.append(counts[0]/counts.sum())

    # pair-wise correlation between features
    correlation = np.corrcoef(features, rowvar=False)
    pooled_correlations.extend(np.triu(correlation, k=1))
    
    # fit logistic regression with linear and quadratic features and store bics
    poly_degree=3
    try:
        # some preprocessing
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(features)
        X = scaler.transform(features)
        X = SelectKBest(f_classif, k=4).fit_transform(X, targets)
        y = targets
        X_linear = PolynomialFeatures(1).fit_transform(X)
        X_poly = PolynomialFeatures(poly_degree).fit_transform(X)   
        log_reg = sm.Logit(y, X_linear).fit(method='bfgs', maxiter=10000)
        log_reg_quadratic = sm.Logit(y, X_poly).fit(method='bfgs', maxiter=10000)
        all_bics_linear.append(log_reg.bic)
        all_bics_quadratic.append(log_reg_quadratic.bic)
    except Exception as e:
        print(e)
        continue
    
    # some preprocessing
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(features)
    inputs = scaler.transform(features)
    inputs = SelectKBest(f_classif, k=4).fit_transform(inputs, targets)
    # shuffle inputs and targets
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    inputs = inputs[indices]
    targets = targets[indices]
    num_trials = 50
    trial = 0 # fit datapoints upto upto_trial; sort of burn-in trials
    baseline_model_choices, true_choices, baseline_model_scores = [], [], []   
    # loop over trials
    while trial < num_trials:
        trial_inputs = inputs[:trial]
        trial_targets = targets[:trial]
        # if all targets until then are same, skip this trial
        if (trial_targets == 0).all() or (trial_targets == 1).all() or trial<=5:
            
            # sample probability from uniform distribution
            p = torch.distributions.uniform.Uniform(0, 1).sample()
            lr_model_choice = torch.tensor([[1-p, p]])
            p = torch.distributions.uniform.Uniform(0, 1).sample()
            svm_model_choice = torch.tensor([[p, 1-p]])
            baseline_model_choices.append(torch.stack([lr_model_choice, svm_model_choice]))
            true_choices.append(torch.tensor(targets[[trial]]))
            baseline_model_scores.append(torch.tensor([p, 1-p]))
        
        else:

            lr_model = LogisticRegressionModel(trial_inputs, trial_targets)
            svm_model = SVMModel(trial_inputs, trial_targets)
            lr_score = lr_model.score(inputs[[trial]], targets[[trial]])
            svm_score = svm_model.score(inputs[[trial]], targets[[trial]])
            lr_model_choice = lr_model.predict_proba(inputs[[trial]])
            svm_model_choice = svm_model.predict_proba(inputs[[trial]])#
            true_choice = torch.tensor(targets[[trial]]) #trial:trial+1]
            baseline_model_choices.append(torch.tensor(np.array([lr_model_choice, svm_model_choice])))
            true_choices.append(true_choice)
            baseline_model_scores.append(torch.tensor(np.array([lr_score, svm_score])))
        trial += 1
    
    # calculate accuracy
    baseline_model_choices_stacked, true_choices_stacked = torch.stack(baseline_model_choices).squeeze(2).argmax(2), torch.stack(true_choices).squeeze()
    accuracy_per_task_lm = (baseline_model_choices_stacked[:, 0] == true_choices_stacked) #for model_id in range(1)]
    accuracy_per_task_svm = (baseline_model_choices_stacked[:, 1] == true_choices_stacked) #for model_id in range(1)]
    
    baseline_model_scores_stacked = torch.stack(baseline_model_scores).squeeze()
    scores.append(baseline_model_scores_stacked.squeeze())
    accuracy_lm.append(accuracy_per_task_lm)
    accuracy_svm.append(accuracy_per_task_svm)

    # # plot target balance
    # fig, axs = plt.subplots(1, 1,  figsize=(5,5))
    # sns.barplot(x=unique, y=counts, ax=axs)
    # plt.tight_layout()
    # sns.despine()
    # plt.show()
    # fig.savefig(f'../figures/target_balance_{task_id}.png')
    
    # # plot pooled pair-wise correlation between features
    # f, ax = plt.subplots(1, 1, figsize=(5,5))
    # sns.histplot(np.triu(correlation, k=1).reshape(-1), ax=ax, bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color='#173b4f')
    # ax.set_title(f'Correlation between features for task {task_id}')
    # f.savefig(f'../figures/correlation_{task_id}.png')

# compute posterior probabilities
logprobs = torch.from_numpy(-0.5 * np.stack((all_bics_linear, all_bics_quadratic), -1))
joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[1])
marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
posterior_logprob = joint_logprob - marginal_logprob


 
# plot trial by trial accuracy
COLORS = {}
f, ax = plt.subplots(1, 1, figsize=(5,5))   
num_tasks = len(accuracy_lm)
plot_last_trials = num_trials
x_labels = np.arange(0, plot_last_trials)
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
f.savefig('../figures/trial_by_trial_accuracy_openML.png')


# plot pooled pair-wise correlation between features, posterior probabilities of linear and quadratic models
fig, axs = plt.subplots(1, 3,  figsize=(15, 5))
sns.histplot(np.concatenate(pooled_correlations), ax=axs[0], bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color=color)
sns.histplot(np.stack(target_balance_ratio), ax=axs[1], bins=5, stat='probability', edgecolor='w', linewidth=1, color=color)
sns.histplot(posterior_logprob[:, 0].exp().detach(), ax=axs[2], bins=5, stat='probability', edgecolor='w', linewidth=1, color=color)
#axs[2].set_ylim(0, 0.5)
axs[0].set_xlim(-1, 1)
axs[0].set_ylim(0, 0.25)
# axs[1].set_ylim(0, 0.3)
axs[0].set_yticks(np.arange(0, 0.25, 0.05))
# axs[1].set_yticks(np.arange(0, 0.35, 0.05))
# set tick size
axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
axs[0].set_ylabel('Percentage', fontsize=FONTSIZE)
axs[1].set_ylabel('')
axs[2].set_ylabel('')
axs[0].set_xlabel('Input correlation', fontsize=FONTSIZE)
axs[1].set_xlabel('Target Balance', fontsize=FONTSIZE)
axs[2].set_xlabel('Linearity', fontsize=FONTSIZE)
plt.tight_layout()
sns.despine()
plt.show()
plt.tight_layout()
sns.despine()
plt.show()
fig.savefig('../figures/data_stats_openML.png')



# plot pooled pair-wise correlation between features, posterior probabilities of linear and quadratic models
fig, axs = plt.subplots(1, 3,  figsize=(15, 5))
sns.histplot(np.concatenate(pooled_correlations), ax=axs[1], bins=11, binrange=(-1., 1.), stat='probability', edgecolor='w', linewidth=1, color=color)
axs[0].plot(x_labels, mean_lm, label='Logistic Regression', color=COLORS['lr'])
axs[0].fill_between(x_labels, mean_lm-std_lm, mean_lm+std_lm, color=COLORS['lr'], alpha=0.2)
axs[0].plot(x_labels, mean_svm, label='SVM', color=COLORS['svm'])
axs[0].fill_between(x_labels, mean_svm-std_svm, mean_svm+std_svm, color=COLORS['svm'], alpha=0.2)
axs[0].hlines(0.5, 0, plot_last_trials, color='k', linestyles='dotted', lw=4)
sns.histplot(posterior_logprob[:, 0].exp().detach(), ax=axs[2], bins=5, stat='probability', edgecolor='w', linewidth=1, color=color)

axs[1].set_xlim(-1, 1)
axs[1].set_ylim(0, 0.25)
axs[1].set_yticks(np.arange(0, 0.25, 0.05))
# set tick size
axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
axs[1].set_ylabel('Percentage', fontsize=FONTSIZE)
axs[2].set_ylabel('')
axs[1].set_xlabel('Input correlation', fontsize=FONTSIZE)
axs[2].set_xlabel('Linearity', fontsize=FONTSIZE)
axs[0].set_xlabel('Trial', fontsize=FONTSIZE)
axs[0].set_ylabel('Mean accuracy (over tasks)', fontsize=FONTSIZE)
axs[0].legend(fontsize=FONTSIZE-2, loc='lower right', frameon=False)
plt.tight_layout()
sns.despine()
plt.show()
plt.tight_layout()
sns.despine()
plt.show()
fig.savefig('../figures/data_stats_openML_2.png')