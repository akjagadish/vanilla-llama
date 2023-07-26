import numpy as np
import pandas as pd
import jsonlines
import ipdb
import anthropic
from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()

datasets = ["data/human/exp1.csv"]
all_prompts = []
categories = {'j': 'A', 'f': 'B'}

for dataset in datasets:
    df = pd.read_csv(dataset)
    df['llm_category'], df['true_category'] = np.nan, np.nan # add new column to df to store the llm predicted category
    num_participants = df.participant.max() + 1
    num_tasks = df.task.max() + 1
    num_blocks = df.block.max() + 1
    
    for participant in range(num_participants):
        df_participant = df[(df['participant'] == participant)]
        num_trials = df_participant.trial.max() + 1 # participant specific number of trials
        num_features = 3 #
        
        # instructions
        instructions =   'In this experiment, you will be shown examples of geometric objects. \n'\
        f'Each object has {num_features} different features: size, color, and shape. \n'\
        'Your job is to learn a rule based on the object features that allows you to tell whether each example \n'\
        f'belongs in the A or B category. As you are shown each \n'\
        'example, you will be asked to make a category judgment and then you \n'\
        'will receive feedback. At first you will have to guess, but you will \n'\
        'gain experience as you go along. Try your best to gain mastery of the \n'\
        'A and B categories. \n\n'\

        for task in range(num_tasks):
            df_task = df_participant[(df_participant['task'] == task)]

            for block in range(num_blocks):
                df_block = df_task[(df_task['block'] == block)]
                num_trials_block = df_block.trial.max() + 1 # block specific number of trials
                block_instructions = instructions #+ f'In this block {block+1}, you will be shown {num_trials_block} examples of geometric objects. \n'

                for t_idx, trial in enumerate(df_block.trial.values[:num_trials_block]):
                    df_trial = df_block[(df_block['trial'] == trial)]
                    t = categories[df_trial.correct_choice.item()]
                    object_name = df_trial.object.item()
                    
                    # anthropic prompt
                    Q_ = anthropic.HUMAN_PROMPT
                    A_ = anthropic.AI_PROMPT
                    question = f'{Q_} What category would a ' + object_name + ' belong to? (Give the answer in the form \"Category <your answer>\").'\
                            f'{A_} Category'
                    query = block_instructions + question
                    print(query)

                    # anthropic api call
                    client = anthropic.Anthropic()
                    response = client.completions.create(
                            prompt = query,
                            model="claude-2",
                            temperature=0.,
                            max_tokens_to_sample=1,
                        ).completion.replace(' ', '')
                    
                    # add llm predicted category and true category to df
                    df.loc[(df['participant'] == participant) & (df['task'] == task) & (df['trial'] == trial) & (df['block'] == block) , 'llm_category'] = response
                    df.loc[(df['participant'] == participant) & (df['task'] == task) & (df['trial'] == trial) & (df['block'] == block) , 'true_category'] = str(t)
                    
                    # add to block instructions
                    block_instructions += '- In trial '+ str(t_idx+1) +', you picked category ' + str(response) + ' for ' + object_name + ' and category ' + str(t) + ' was correct.\n'
    
# save df with llm predicted category and true category
df.to_csv(dataset.replace('.csv', '_llm.csv'), index=False)
