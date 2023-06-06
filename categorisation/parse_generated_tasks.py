import pickle
import re
import pandas as pd
import numpy as np

file_name='llama_generated_tasks_65B_3'
# load llama generated tasks which were successfully regex parsed
with open(f"data/{file_name}.txt", "rb") as fp:   
    datasets = pickle.load(fp)

# regular expression pattern to extract input values from the stored inputs and targets
pattern = r'((?: )?\s*[\d.]+)(?:,|;|\s)?\s*([\d.]+)(?:,|;|\s)?\s*([\d.]+)'
# ((?:\s|\[)?\s*[\d.]+) (?:,|;|\s) ?\s*([\d.]+) (?:,|;|\s) ?\s*([\d.]+)'

# make a pandas dataframe for the parsed data
df = df.read_csv('data/{file_name}.csv') if else None
task_id=1
# parse the list using regex
for task, data in enumerate(datasets):
    # initialize lists to store parsed values
    inputs = []
    targets = []
    for item in data:
        match = re.match(pattern, item[0])
        if match:
            try:
                inputs.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
                targets.append(item[1])
            except:
                print(f'error parsing {task, item[0]}')
        else:
            print(f'error parsing {task, item[0]}')
    if len(inputs) == 8:
        # create a DataFrame from inputs and targets
        df = pd.DataFrame({'input': inputs, 'target': targets, 'task_id': np.ones((len(inputs),))*(task_id)}) if df is None else pd.concat([df, \
        pd.DataFrame({'input': inputs, 'target': targets, 'task_id': np.ones((len(inputs),))*(task_id)})], ignore_index=True)
        task_id+=1

# print the DataFrame
print(df)
#save data frame to csv
df.to_csv('data/llama_generated_tasks_65B_3.csv')