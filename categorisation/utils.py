import pickle
import re
import pandas as pd
import numpy as np
import sys
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/data')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/rl2')
from baseline_classifiers import benchmark_baseline_models_regex_parsed_random_points, benchmark_baseline_models_regex_parsed

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
                inputs.append([float(item[0]), float(item[1]), float(item[2])])
                targets.append(item[3])
                
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
        if gpt=='gpt3' or ((gpt=='llama') and (len(inputs)==num_datapoints)):
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
