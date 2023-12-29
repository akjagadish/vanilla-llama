import pickle
import re
import pandas as pd
import numpy as np
import torch
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
sys.path.append("/u/ajagadish/vanilla-llama/categorisation/rl2")
sys.path.append("/u/ajagadish/vanilla-llama/categorisation/")
sys.path.append("/u/ajagadish/vanilla-llama/categorisation/data")
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/data')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/rl2')
from baseline_classifiers import benchmark_baseline_models_regex_parsed_random_points, benchmark_baseline_models_regex_parsed
from baseline_classifiers import LogisticRegressionModel, SVMModel


def parse_generated_tasks(path, file_name, gpt, num_datapoints=8, num_dim=3, last_task_id=0, use_generated_tasklabels=False, prompt_version=None):
   
    # load llama generated tasks which were successfully regex parsed
    with open(f"{path}/{file_name}.txt", "rb") as fp:   
        datasets = pickle.load(fp)

    # regular expression pattern to extract input values from the stored inputs and targets
    pattern = r'((?: )?\s*[\d.]+)(?:,|;|\s)?\s*([\d.]+)(?:,|;|\s)?\s*([\d.]+)'

    # make a pandas dataframe for the parsed data
    df = None 
    task_id = last_task_id

    # load task labels if use_generated_tasklabels is True
    if use_generated_tasklabels:
        with open(f"{path}/{file_name}_taskids.txt", "rb") as fp:   
            task_label = pickle.load(fp)

    # parse the list using regex
    for task, data in enumerate(datasets):
        # initialize lists to store parsed values
        inputs, targets = [], []
        #TODO: per data make the target into A or B

        # load each input-target pair
        for item in data:
            if gpt == 'gpt3':
    
                inputs.append([float(item[0]), float(item[1]), float(item[2])])
                targets.append(item[3])

            elif gpt == 'gpt4':
       
                inputs.append([float(item[0]), float(item[1]), float(item[2])])
                targets.append(item[3][1] if len(item[3])>1 else item[3])

            elif gpt == 'claude':
                
                if num_dim==3:
                    if prompt_version == 3:
                        inputs.append([item[0], item[1], item[2]])
                    elif prompt_version == 4:
                        try:
                            inputs.append([float(item[0]), float(item[1]), float(item[2])])
                        except:
                            print(f'{task} not parsable as float')
                            continue
                    
                elif num_dim==6:
                    if prompt_version == 5:
                        try:    
                            inputs.append([float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]), float(item[6])])
                        except:
                            print(f'{task} not parsable as float')
                            continue
                    else:
                        raise NotImplementedError
                
                elif num_dim==4:
                    
                    if prompt_version == 5:
                        try:    
                            inputs.append([float(item[1]), float(item[2]), float(item[3]), float(item[4])])
                        except:
                            print(f'{task} not parsable as float')
                            continue
                    else:
                        raise NotImplementedError

                else:
                    raise NotImplementedError
                    
                if use_generated_tasklabels:
                    targets.append(item[num_dim+1])
                else:
                    targets.append(item[3][1] if len(item[3])>1 else item[3])
                
            else:
                match = re.match(pattern, item[0])

                if match:
                    try:
                        inputs.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
                        targets.append(item[1])
                    except:
                        print(f'no match')
                else:
                    print(f'error parsing {task, item[0]}')

        # if the number of datapoints is equal to the number of inputs, add to dataframe
        if gpt=='gpt3' or gpt=='gpt4' or gpt=='claude' or ((gpt=='llama') and (len(inputs)==num_datapoints)):
            print(f'{task} has inputs of length {len(inputs)}')
            use_task_index = task_label[task] if use_generated_tasklabels else task_id
            df = pd.DataFrame({'input': inputs, 'target': targets, 'trial_id': np.arange(len(inputs)), 'task_id': np.ones((len(inputs),))*(use_task_index)}) if df is None else pd.concat([df, \
                 pd.DataFrame({'input': inputs, 'target': targets, 'trial_id': np.arange(len(inputs)), 'task_id': np.ones((len(inputs),))*(use_task_index)})], ignore_index=True)
            task_id+=1
        else:
            print(f'dataset did not have {num_datapoints} datapoints but instead had {len(inputs)} datapoints')

    # save data frame to csv
    if df is not None:
        df.to_csv(f'{path}/{file_name}.csv')
    else:
        print(f'no datasets were successfully parsed')

    return task_id

def return_generated_task(path, gpt, model, num_dim, num_data, num_tasks, run, proc_id, prompt_version, stage):
    filename = f'{gpt}_generated_tasks_params{model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}'
    if stage>=1:
        filename = f'{filename}_stage{stage}'
    return pd.read_csv(f"{path}/{filename}.csv")

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

# computing the distance between consequetive datapoints over trials
def l2_distance_trials(data, within_targets=False, within_consecutive_targets=False):
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
            
            if within_consecutive_targets:
                # consequetive datapoints with the same target
                distance = np.array([np.linalg.norm(inputs[ii]-inputs[ii+1]) for ii in range(inputs.shape[0]-1) if targets[ii]==targets[ii+1]])
            
            # pad with Nan's if distances are of unequal length and stack them vertically over tasks
            distance = np.pad(distance, (0, int(data.trial_id.max()*0.6)-distance.shape[0] if within_targets else data.trial_id.max()-distance.shape[0]), mode='constant', constant_values=np.nan)
            if task==0:
                distances = distance
            else:
                distances = np.vstack((distances, distance))

        # # plot the spatial distances
        # f, ax = plt.subplots(1, 1, figsize=(7,7))   
        # sns.heatmap(distances, annot=False, ax=ax, cmap='hot_r', vmin=0, vmax=1)
        # ax.set_title(f'Distance between datapoints')
        # ax.set_xlabel('Trial')
        # ax.set_ylabel('Task')
        # plt.show()
    
    return distances

def l2_distance_trials_all(data, target='A', shift=1, within_targets=False, llama=False, random=False):
    '''
    Compute distance of a datapoint with every other datapoint with shifts over trials
    Args:
        data: pandas dataframe with columns ['task_id', 'trial_id', 'input', 'target']
    Returns:
        None
    ''' 
    tasks = data.task_id.unique()#[:1000]

    # extract the distances for each task
    for task in tasks:
        
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        targets = np.stack([val for val in data[data.task_id==task].target.values]) 
        
        if random:
            num_points, dim = 87, 3
            inputs =  np.random.rand(num_points, dim) 
            targets = np.random.choice(['A', 'B'], size=num_points) 

        if within_targets:
            inputs = inputs[targets==target]    

        # get the spatial distance between datapoints over trials 
        distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[ii+shift,:]) for ii in range(inputs.shape[0]-shift)])
        # pad with Nan's if distances are of unequal length and stack them vertically over tasks
        if llama:
            distance = np.pad(distance, (0, int(8*0.6)-distance.shape[0] if within_targets else 8-distance.shape[0]), mode='constant', constant_values=np.nan)
        else:
            distance = np.pad(distance, (0, int(data.trial_id.max()*0.9)-distance.shape[0] if within_targets else data.trial_id.max()+1-distance.shape[0]), mode='constant', constant_values=np.nan)

        if task==0:
            distances = distance
        else:
            distances = np.vstack((distances, distance))
        
    return distances

def probability_same_target_vs_distance(data, target='A', llama=False, random=False):

    #TODO:
    # 1. set max datapoints based on max number of trials in the dataset
    # 2. set values for random more appropriately
    ref_target = 0 if target=='A' else 1
    tasks = data.task_id.unique()#[:1000]
    MAX_SIZE = (data.trial_id.max()+1)**2
    # load data for each task
    for task in tasks:
        
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack([eval(val) for val in data[data.task_id==task].input.values])
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        targets = np.stack([val for val in data[data.task_id==task].target.values]) 
        
        if random:
            num_points, dim = data.trial_id.max(), 3
            inputs =  np.random.rand(num_points, dim) 
            targets = np.random.choice(['A', 'B'], size=num_points) 
   

        # get the spatial distance between datapoints over trials 
        #feature_distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[ii+1,:]) for ii in range(inputs.shape[0]-1)])
        # distance between every pair of datapoints
        feature_distance = np.array([np.linalg.norm(inputs[ii,:]-inputs[jj,:]) for ii in range(inputs.shape[0]) for jj in range(inputs.shape[0]) if ii!=jj])

        # compute difference in probability of target for each pair of datapoints
        svm = SVMModel(inputs, targets)
        probability = svm.predict_proba(inputs)
        #probability_distance = np.array([np.linalg.norm(probability[ii, 0]-probability[ii+1, 0]) for ii in range(probability.shape[0]-1)])
        probability_distance = np.array([np.abs(probability[ii,ref_target]-probability[jj,ref_target]) for ii in range(probability.shape[0]) for jj in range(probability.shape[0]) if ii!=jj])
        
        # pad with Nan's if distances are of unequal length and stack them vertically over tasks
        # 100*100 = 10000 is the maximum number of pairs of datapoints as number of datapoints is 100
        probability_distance = np.pad(probability_distance, (0, MAX_SIZE-feature_distance.shape[0]), mode='constant', constant_values=np.nan)
        feature_distance = np.pad(feature_distance, (0, MAX_SIZE-feature_distance.shape[0]), mode='constant', constant_values=np.nan)
        
        #print(probability_distance.shape)
        if task==0:
            distances = feature_distance
            probabilities = probability_distance
        else:
            distances = np.vstack((distances, feature_distance))
            probabilities = np.vstack((probabilities, probability_distance))

    # # plot probability vs distance
    # f, ax = plt.subplots(1, 1, figsize=(7,7))
    # sns.regplot(distances.flatten(), probabilities.flatten(), ax=ax)
    # ax.set_title(f'Probability of same target vs distance between datapoints')
    # ax.set_xlabel('Distance between datapoints')
    # ax.set_ylabel('Probability of same target')
    # plt.show()

    return distances, probabilities

def evaluate_data_against_baselines(data, upto_trial=15, num_trials=None):

    tasks = data.task_id.unique()#[:1000] 
    accuracy_lm = []
    accuracy_svm = []
    scores = []
    # loop over dataset making predictions for next trial using model trained on all previous trials
    for task in tasks:
        baseline_model_choices, true_choices, baseline_model_scores = [], [], []   
        # get the inputs for this task which is numpy array of dim (num_trials, 3)
        inputs = np.stack(data[data.task_id==task].input.values)
        # normalise data for each task to be between 0 and 1
        inputs = (inputs - inputs.min())/(inputs.max() - inputs.min())
        # get the targets for this task which is numpy array of dim (num_trials, 1)
        # targets = torch.stack([torch.tensor(0) if val=='A' else torch.tensor(1) for val in data[data.task_id==task].target.values])
        targets = data[data.task_id==task].target.to_numpy()
        targets = torch.from_numpy(np.unique(targets, return_inverse=True)[1])
        num_trials = data[data.task_id==task].trial_id.max() if num_trials is None else num_trials

        trial = upto_trial # fit datapoints upto upto_trial; sort of burn-in trials
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
                true_choices.append(targets[[trial]])
                baseline_model_scores.append(torch.tensor([p, 1-p]))
            
            else:

                lr_model = LogisticRegressionModel(trial_inputs, trial_targets)
                svm_model = SVMModel(trial_inputs, trial_targets)
                lr_score = lr_model.score(inputs[[trial]], targets[[trial]])
                svm_score = svm_model.score(inputs[[trial]], targets[[trial]])
                lr_model_choice = lr_model.predict_proba(inputs[[trial]])
                svm_model_choice = svm_model.predict_proba(inputs[[trial]])#
                true_choice = targets[[trial]] #trial:trial+1]
                baseline_model_choices.append(torch.tensor(np.array([lr_model_choice, svm_model_choice])))
                true_choices.append(true_choice)
                baseline_model_scores.append(torch.tensor(np.array([lr_score, svm_score])))
            trial += 1
    
        # calculate accuracy
        baseline_model_choices_stacked, true_choices_stacked = torch.stack(baseline_model_choices).squeeze().argmax(2), torch.stack(true_choices).squeeze()
        accuracy_per_task_lm = (baseline_model_choices_stacked[:, 0] == true_choices_stacked) #for model_id in range(1)]
        accuracy_per_task_svm = (baseline_model_choices_stacked[:, 1] == true_choices_stacked) #for model_id in range(1)]
        
        baseline_model_scores_stacked = torch.stack(baseline_model_scores).squeeze()
        scores.append(baseline_model_scores_stacked.squeeze())
        accuracy_lm.append(accuracy_per_task_lm)
        accuracy_svm.append(accuracy_per_task_svm)
        
    return accuracy_lm, accuracy_svm, scores

def find_counts(inputs, dim, xx_min, xx_max):
    return (inputs[:, dim]<xx_max)*(inputs[:, dim]>xx_min)

def data_in_range(inputs, targets, min_value=0, max_value=1):
    inputs_in_range = [(inputs[ii]>min_value).all() * (inputs[ii]<max_value).all() for ii in range(len(inputs))]
    inputs = inputs[inputs_in_range]
    targets = targets[inputs_in_range]
    return inputs, targets

def bin_data_points(num_bins, data, min_value=0, max_value=1):
    inputs = np.stack([eval(val) for val in data.input.values])
    targets = np.stack([val for val in data.target.values])
    inputs, targets = data_in_range(inputs, targets, min_value, max_value)
    bins = np.linspace(0, 1, num_bins+1)[:-1]
    bin_counts, target_counts = [], [] #np.zeros((len(bins)*3))
    for ii in bins:
        x_min = ii 
        x_max = ii + 1/num_bins
        for jj in bins:
            y_min = jj
            y_max = jj + 1/num_bins
            for kk in bins:
                z_min = kk
                z_max = kk + 1/num_bins
                num_points = (find_counts(inputs, 0, x_min, x_max)*find_counts(inputs, 1, y_min, y_max)*find_counts(inputs, 2, z_min, z_max))
                bin_counts.append(num_points.sum())
                target_counts.append((targets[num_points]=='A').sum())

    bin_counts = np.array(bin_counts)
    target_counts = np.array(target_counts)
    return bin_counts, target_counts

def gini_compute(x):
                    mad = np.abs(np.subtract.outer(x, x)).mean()
                    rmad = mad/np.mean(x)
                    return 0.5 * rmad

def return_data_stats(data, poly_degree=2):

    df = data.copy()
    max_tasks = int(df['task_id'].max() + 1)
    all_corr, all_coef, all_bics_linear, all_bics_quadratic  = [], [], [], []
    f1_ceof, f2_coef, f3_coef = [], [], []
    f1_corr, f2_corr, f3_corr = [], [], []
    gini_coeff, advantage = [], []
    for i in range(0, max_tasks):
        df_task = df[df['task_id'] == i]
        if len(df_task) > 50: # arbitary data size threshold
            y = df_task['target'].to_numpy()
            y = np.unique(y, return_inverse=True)[1]

            # df_task['input'] = df_task['input'].apply(eval).apply(np.array)
            X = df_task["input"].to_numpy()
            X = np.stack(X)
            X = (X - X.min())/(X.max() - X.min())  # normalize data
            
            # correlations
            all_corr.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
            all_corr.append(np.corrcoef(X[:, 0], X[:, 2])[0, 1])
            all_corr.append(np.corrcoef(X[:, 1], X[:, 2])[0, 1])

            # per feature correlations
            f1_corr.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
            f2_corr.append(np.corrcoef(X[:, 0], X[:, 2])[0, 1])
            f3_corr.append(np.corrcoef(X[:, 1], X[:, 2])[0, 1])

            if (y == 0).all() or (y == 1).all():
                pass
            else:
                X_linear = PolynomialFeatures(1).fit_transform(X)
                log_reg = sm.Logit(y, X_linear).fit(method='bfgs', maxiter=10000)

                # weights
                all_coef.append(log_reg.params[1])
                all_coef.append(log_reg.params[2])
                all_coef.append(log_reg.params[3])
                
                # store feature specific weights separately
                f1_ceof.append(log_reg.params[1])
                f2_coef.append(log_reg.params[2])
                f3_coef.append(log_reg.params[3])


                X_poly = PolynomialFeatures(poly_degree).fit_transform(X)
                log_reg_quadratic = sm.Logit(y, X_poly).fit(method='bfgs', maxiter=10000)
                
                # svm using sklearn
                # svm = SVMModel(X, y)
                # score_svm = svm.score(X, y)
                # bic_svm, ll_svm = svm.calculate_bic(X, y)
                
                # logisitic regression with polynomial features using sklearn
                svm = LogisticRegressionModel(X_poly, y)
                score_svm = svm.score(X_poly, y)
                bic_svm, ll_svm = svm.calculate_bic(X_poly, y)
                
                # logisitic regression with linear features using sklearn
                lr = LogisticRegressionModel(X_linear, y)
                score_lr = lr.score(X_linear, y)
                bic_lr, ll_lr = lr.calculate_bic(X_linear, y)
                # print(bic_lr, bic_svm)
               
                advantage.append(0. if score_svm > score_lr else 1.) #0 if score_svm > score_lr else 1) #(score_svm - score_lr) #(ll_svm-ll_lr)

                # bics
                all_bics_linear.append(bic_lr) #(-2*ll_lr) #(log_reg.bic)
                all_bics_quadratic.append(bic_svm) #(-2*ll_svm) #(log_reg_quadratic.bic)

                gini = gini_compute(np.abs(log_reg.params[1:])) 
                gini_coeff.append(gini)
                

    # compute posterior probabilities
    logprobs = torch.from_numpy(-0.5 * np.stack((all_bics_linear, all_bics_quadratic), -1))
    joint_logprob = logprobs + torch.log(torch.ones([]) /logprobs.shape[1])
    marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
    posterior_logprob = joint_logprob - marginal_logprob

    # store feature specific weights separately
    f1_ceof = np.array(f1_ceof)
    f2_coef = np.array(f2_coef)
    f3_coef = np.array(f3_coef)
    feature_coef = np.stack((f1_ceof, f2_coef, f3_coef), -1)
    # horizontal task the feature coefficients
    # hfeature_coef = feature_coef.reshape(-1, 3)

    # horizontal stack per feature correlations
    features_corrs = np.stack((f1_corr, f2_corr, f3_corr), -1)


    return all_corr, all_coef, posterior_logprob, feature_coef, features_corrs, gini_coeff, advantage

def retrieve_features_and_categories(path, file_name, task_id):
 
    df = pd.read_csv(f'{path}/{file_name}.csv')
    task_id = df.task_id[task_id]
    df = df[df.task_id==df.task_id[task_id]]
    features = eval(df.feature_names.values[0])
    categories = eval(df.category_names.values[0])
    return features, categories, task_id

def pool_tasklabels(path_to_dir, run_gpt, model, num_dim, num_tasks, num_runs, proc_id, prompt_version, num_categories=2):
    df, last_task_id = None, 0
    for run_id in range(num_runs):
        data = None
        try:
            filename = f'{run_gpt}_generated_tasklabels_params{model}_dim{num_dim}_tasks{num_tasks}_run{run_id}_procid{proc_id}_pversion{prompt_version}'
            data = pd.read_csv(f'{path_to_dir}/{filename}.csv')       
        except:
            print(f'error loading {filename}')
        if data is not None:
            # does number of features match the number of dimensions
            features = [eval(feature) for feature in data.feature_names.values]
            features_match = np.array([len(feature) for feature in features])==num_dim 
            # does number of categories match the number of dimensions
            categories = [eval(category) for category in data.category_names.values]
            categories_match = np.array([len(category) for category in categories])==num_categories
            # if both match, add to dataframe
            both_match = features_match*categories_match
            processed_data = pd.DataFrame({'feature_names': data.feature_names.values[both_match], 'category_names': data.category_names.values[both_match], 'task_id': np.arange(len(data.task_id.values[both_match])) + last_task_id})
            df = processed_data if df is None else pd.concat([df, processed_data], ignore_index=True)
            last_task_id = df.task_id.values[-1] + 1


    num_tasks = df.task_id.max()+1
    # df.feature_names = df['feature_names'].apply(lambda x: eval(x))
    # df.category_names = df['category_names'].apply(lambda x: eval(x))
    df.to_csv(f'{path_to_dir}/{run_gpt}_generated_tasklabels_params{model}_dim{num_dim}_tasks{num_tasks}_pversion{prompt_version}.csv')             

def get_regex_patterns(num_dim, use_generated_tasklabels, prompt_version):
    ''' 
    Generate regex patterns to parse the generated tasks
    Args:
        num_dim: number of dimensions
        use_generated_tasklabels: whether to use the generated tasklabels or not
        prompt_version: version of the prompt used to generate the tasks
    Returns:
        patterns: list of regex patterns
    '''
    if use_generated_tasklabels is False:
        og_regex_expressions = [r'x=\[(.*?)\][;,]?\s*y\s*=?\s*([AB])',
                        r"x=\[(.*?)\][^\n]*?y=(\w)",
                        r"x=\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\][^\n]*[y|&]=\s*(A|B)",
                        r"x=\[?\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]?\s*(?:,|;|&|->| -> |---)?\s*[y|Y]\s*=\s*(A|B)",
                        r"x=(\[.*?\])\s*---\s*y\s*=\s*([A-Z])",
                        r"x=(\[.*?\])\s*->\s*([A-Z])",
                        r"x=(\[.*?\]),\s*([A-Z])",
                        r"^([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2}),(A|B)$",
                        r"\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(A|B)",
                        r"\[\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(A|B)\]",
                        r"n[0-9]+\.\[\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(\'A\'|\'B\')\]",
                        r"\[\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(\'A\'|\'B\')\]",
                        r"\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(A|B)",
                        r"(\d+\.\d+),(\d+\.\d+),(\d+\.\d+),([A-Z])"
                        ] 
    elif num_dim == 3 and prompt_version == 4: 
        regex_expressions = [r'([\d.]+),([\d.]+),([\d.]+),([\w]+)',
                r'([\w\-]+),([\w\-]+),([\w\-]+),([\w]+)',
                r'([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+)',
                r'([^,]+),([^,]+),([^,]+),([^,]+)',
                r'([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+)',
                r'(?:.*?:)?([^,-]+),([^,-]+),([^,-]+),([^,-]+)',
                r'([^,-]+),([^,-]+),([^,-]+),([^,-]+)',]
                        
    elif num_dim == 6 and prompt_version == 5: 
        regex_expressions = [r'^(\d+):([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\w]+)',
                r'^(\d+):([\w\-]+),([\w\-]+),([\w\-]+),([\w\-]+),([\w\-]+),([\w\-]+),([\w]+)',
                r'^(\d+):([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+)',
                r'^(\d+):([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)',
                r'^(\d+):([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+)',
                r'^(\d+):(?:.*?:)?([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+)',
                r'^(\d+):([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+)',]

    elif num_dim == 4 and prompt_version == 5:
        regex_expressions = [r'^(\d+):([\d.]+),([\d.]+),([\d.]+),([\d.]+),([\w]+)',
                r'^(\d+):([\w\-]+),([\w\-]+),([\w\-]+),([\w\-]+),([\w]+)',
                r'^(\d+):([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+)',
                r'^(\d+):([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)',
                r'^(\d+):([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+)',
                r'^(\d+):(?:.*?:)?([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+)',
                r'^(\d+):([^,-]+),([^,-]+),([^,-]+),([^,-]+),([^,-]+)']
                         
    patterns = regex_expressions if use_generated_tasklabels else og_regex_expressions

    return patterns           