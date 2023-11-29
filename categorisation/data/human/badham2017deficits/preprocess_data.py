import pandas as pd
import numpy as np

# Define the conditions: convert from string to integer
CONDITIONS = {'TYPEONE': 1, 'TYPETWO': 2, 'TYPETHREE': 3, 'TYPEFOUR': 4}
## Feature Coding Big: 1, Small: 0, Black: 1, White: 0, Square: 1, Triangle: 0; in same order as the object name
FEATURES = {'BigBlackSquare': [1, 1, 1], 'SmallBlackTriangle': [0, 1, 0],
            'BigWhiteTriangle': [1, 0, 0], 'SmallWhiteTriangle': [0, 0, 0],
            'SmallWhiteSquare': [0, 0, 1], 'SmallBlackSquare': [0, 1, 1],
            'BigWhiteSquare': [1, 0, 1], 'BigBlackTriangle': [1, 1, 0]}
FEATURE_NAMES_SPLIT = {'BigBlackSquare': 'Big Black Square', 'SmallBlackTriangle': 'Small Black Triangle',
                        'BigWhiteTriangle': 'Big White Triangle', 'SmallWhiteTriangle': 'Small White Triangle',
                        'SmallWhiteSquare': 'Small White Square', 'SmallBlackSquare': 'Small Black Square',
                        'BigWhiteSquare': 'Big White Square', 'BigBlackTriangle': 'Big Black Triangle'}
# Read the data
data = pd.read_csv('Badham_Sanborn_Maylor_2017_trial_by_trial_data.csv')
data = data.rename(columns={"Slide1.RESP": "cond1_response", "Slide2.RESP": "cond2_response",\
                            "Slide3.RESP": "cond3_response", "Slide4.RESP": "cond4_response", \
                            "Slide1.CRESP": "correct_cond1_response", "Slide2.CRESP": "correct_cond2_response",\
                            "Slide3.CRESP": "correct_cond3_response", "Slide4.CRESP": "correct_cond4_response", \
                            "Slide1.ACC": "cond1_acc", "Slide2.ACC": "cond2_acc",\
                            "Slide3.ACC": "cond3_acc", "Slide4.ACC": "cond4_acc", \
                            })

# Create a new dataframe
num_subjects = data.Subject.nunique()
num_conditions = 4 # four types of categorisation tasks
num_blocks = 4 # four blocksm, one per condition
new_df = pd.DataFrame(columns=['participant', 'task', 'trial', 'choice', 'correct_choice', 'reward', 'block', 'condition', 'category',  'object', 'all_features', 'feature1', 'feature2', 'feature3'])
subject_id = 0 # to keep track of the subject id for whom the data is formatted correctly

# Loop through each subject and each block
for subject in range(1, num_subjects+1):
    total_trials_until_previous_block = 0 # variable number of trials per block so keep track of the total number of trials
    for block_number in range(1, num_conditions+1):

        try:
            df = data[data.Subject==subject].query(f'Block=={block_number}')
            assert data[data.Subject==subject].Block.nunique() == num_conditions, 'not enough blocks'
            cond_number = CONDITIONS[df['Running[Block]'].values[0]]
            df = df.query(f'cond{cond_number}_response == "j" or cond{cond_number}_response == "f"')
            response = df[f'cond{cond_number}_response'].values
            correct_response =  df[f'correct_cond{cond_number}_response'].values
            correct =  df[f'cond{cond_number}_acc'].values
            # make correct in the format required: 1 or -1
            correct[correct==0] = -1
            trial_ids = np.arange(1, len(response)+1) + total_trials_until_previous_block
            conditions = np.repeat(cond_number, len(response))
            blocks = np.repeat(block_number, len(response))-1

            object_name = np.stack([FEATURE_NAMES_SPLIT[df.FileName.values[ii][:-4]] for ii in range(len(response))])
            feature = np.stack([FEATURES[df.FileName.values[ii][:-4]] for ii in range(len(response))])
            all_features = [str(list(ff)) for ff in feature]
            feature1 = feature[:, 0]
            feature2 = feature[:, 1]
            feature3 = feature[:, 2]
            category = np.where(df['Alpha']=='ALPHA', 0, 1)
            
            new_df = new_df.append(pd.DataFrame({'participant': subject_id, 'task':0, 'block': blocks, 'condition': conditions, \
                                                'trial': trial_ids-1, 'category': category, 'reward': correct, \
                                                'choice': response, 'correct_choice': correct_response,'object': object_name, 'all_features': all_features, 'feature1': feature1, \
                                            'feature2': feature2, 'feature3': feature3}), ignore_index=True)
            if block_number==num_blocks:
                subject_id += 1 
            total_trials_until_previous_block = trial_ids[-1]
        except:
            print(f'Issue with the data for subject {subject} block {block_number}')
            continue   

# Save the dataframe
new_df.to_csv('./exp1.csv', index=False)