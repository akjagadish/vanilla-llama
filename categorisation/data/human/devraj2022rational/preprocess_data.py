import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

# load csv file but ingore first row as header
experiment_1 = pd.read_csv('./experiment1.csv', header=None)
experiment_1 = experiment_1.rename(columns={0: 'stimulus_sequence'}) # name first column as stimulus sequence

# extract the stimulus sequence for each participant
stimulus_sequence, choices, true_categories, conditions = [], [], [], []
for stim_seq_id in experiment_1.stimulus_sequence.values:
    participant_choices = experiment_1[experiment_1.stimulus_sequence==stim_seq_id].values.squeeze()[1:]
    stimuli = pd.read_csv(f'./stimuli/stim_list_{stim_seq_id}.csv', header=None).values.squeeze()
    participant_true_categories = np.zeros_like(stimuli) + 1
    participant_true_categories[stimuli>=7] = 2
    stimulus_sequence.append(stimuli)
    choices.append(participant_choices)
    true_categories.append(participant_true_categories)
    conditions.append('control' if stim_seq_id%2==0 else 'experimental')

#stimuli 0 to 6: [000000, 100000, 010000, 001000, 000010, 000001, 111101] and 7 to 13: [111111, 011111, 101111, 110111, 111011, 111110, 000010]
stimulus_features = {0: [0, 0, 0, 0, 0, 0], 1: [1, 0, 0, 0, 0, 0], 2: [0, 1, 0, 0, 0, 0], 3: [0, 0, 1, 0, 0, 0], \
                     4: [0, 0, 0, 0, 1, 0], 5: [0, 0, 0, 0, 0, 1], 6: [1, 1, 1, 1, 0, 1], 7: [1, 1, 1, 1, 1, 1], \
                     8: [0, 1, 1, 1, 1, 1], 9: [1, 0, 1, 1, 1, 1], 10: [1, 1, 0, 1, 1, 1], 11: [1, 1, 1, 0, 1, 1],\
                    12: [1, 1, 1, 1, 1, 0], 13: [0, 0, 0, 1, 0, 0]}
prototype_features = {1: [0, 0, 0, 0, 0, 0], 2: [1, 1, 1, 1, 1, 1]}
                     
# make a new dataframe with each row containing features of the task of one trial for a given participant
new_df = pd.DataFrame(columns=['participant', 'task', 'trial', 'choice', 'correct_choice', 'block', 'trial_segment', 'condition', 'category', 'all_features', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6'])
for participant_id in range(len(stimulus_sequence)):
    for trial_id in range(len(stimulus_sequence[participant_id])):
        block_id = trial_id//14
        trial_segment = trial_id//56
        new_df = new_df.append(pd.DataFrame({'participant': [participant_id], 'task': [0], 'block': [block_id], 'trial_segment': [trial_segment], 'condition': [conditions[participant_id]], \
                                             'trial': [trial_id], 'stimulus_id': [stimulus_sequence[participant_id][trial_id]],'category': [true_categories[participant_id][trial_id]], 'choice': [choices[participant_id][trial_id]-1], \
                                             'correct_choice': [true_categories[participant_id][trial_id]-1], 'all_features': [str(stimulus_features[stimulus_sequence[participant_id][trial_id]])], \
                                             'feature1': [stimulus_features[stimulus_sequence[participant_id][trial_id]][0]], 'feature2': [stimulus_features[stimulus_sequence[participant_id][trial_id]][1]], \
                                             'feature3': [stimulus_features[stimulus_sequence[participant_id][trial_id]][2]], 'feature4': [stimulus_features[stimulus_sequence[participant_id][trial_id]][3]], \
                                             'feature5': [stimulus_features[stimulus_sequence[participant_id][trial_id]][4]], 'feature6': [stimulus_features[stimulus_sequence[participant_id][trial_id]][5]], \
                                             'prototype_feature1': [prototype_features[true_categories[participant_id][trial_id]][0]], 'prototype_feature2': [prototype_features[true_categories[participant_id][trial_id]][1]], \
                                             'prototype_feature3': [prototype_features[true_categories[participant_id][trial_id]][2]], 'prototype_feature4': [prototype_features[true_categories[participant_id][trial_id]][3]], \
                                             'prototype_feature5': [prototype_features[true_categories[participant_id][trial_id]][4]], 'prototype_feature6': [prototype_features[true_categories[participant_id][trial_id]][5]]}), ignore_index=True)
                                             

# save the dataframe as a csv file
new_df.to_csv('../devraj2022rational.csv', index=False)

# plot accuracy of choice for each participant over trial segments using new dataframe
accuracy = []
for participant_id in new_df.participant.unique():
    new_df_participant = new_df[new_df.participant==participant_id]
    participant_acc = []
    for trial_segment in new_df_participant.trial_segment.unique():
        new_df_participant_trial_segment = new_df_participant[new_df_participant.trial_segment==trial_segment]
        participant_acc.append(np.sum(new_df_participant_trial_segment.choice.values==new_df_participant_trial_segment.correct_choice.values)/len(new_df_participant_trial_segment))
    accuracy.append(participant_acc)

# plot mean accuracy of participants over trial segments as line plot
f, ax = plt.subplots(1, 1, figsize=(5,5))
sns.lineplot(x=np.arange(11), y=np.mean(accuracy, axis=0), ax=ax)
# add standard error of mean as error bars
ax.fill_between(np.arange(11), np.mean(accuracy, axis=0)-np.std(accuracy, axis=0)/np.sqrt(len(accuracy)), np.mean(accuracy, axis=0)+np.std(accuracy, axis=0)/np.sqrt(len(accuracy)), alpha=0.2)
ax.set_xlabel('Trial segment')
ax.set_ylabel('Accuracy')
ax.set_xticks(np.arange(11))
ax.set_xticklabels(np.arange(11)+1)
ax.set_ylim([0, .95])
ax.set_title('Accuracy over trial segments')
sns.despine()
f.tight_layout()
plt.show()



