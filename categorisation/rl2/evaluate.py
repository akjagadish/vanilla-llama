
import numpy as np
import torch
from envs import CategorisationTask
import argparse
from baseline_classifiers import LogisticRegressionModel, SVMModel

# evaluate a model
def evaluate(env_name=None, model_path=None, env=None, model=None, mode='val', policy='greedy', return_all=False):
    
    if env is None:
        # load environment
        env = CategorisationTask(data=env_name, mode=mode)
    if model is None:
        # load model
        model = torch.load(model_path)[1]
        
    with torch.no_grad():
        model.eval()
        inputs, targets, prev_targets, done, info = env.reset()
        hx, cx = model.initial_states(env.batch_size)
        model_choices = []
        true_choices = []
        
        while not done:
            inputs = model.make_inputs(inputs, prev_targets) 
            model_choice, hx, cx = model(inputs.float(), hx, cx)
            true_choice = targets.detach().clone()
            model_choices.append(model_choice)
            true_choices.append(true_choice)
            inputs, targets, prev_targets, done, info = env.step()
        
        model_choices = torch.stack(model_choices).squeeze()
        true_choices = torch.stack(true_choices)

        predictions = model_choices.argmax(2).reshape(-1) if policy=='greedy' else \
            model_choices.view(model_choices.shape[0]*model_choices.shape[1], model_choices.shape[2]).multinomial(1).reshape(-1)
        accuracy = (true_choices.reshape(-1)==predictions).sum()/(predictions.shape[0])
        
    if return_all:
        return accuracy, model_choices, true_choices
    else:    
        return accuracy
def evaluate_against_baselines(env_name, model_path, mode='val', return_all=False):

    # load environment
    env = CategorisationTask(data=env_name, mode=mode)
    
    # load models
    _, _, _, done, info = env.reset()
    inputs, targets = info['inputs'], info['targets']
    baseline_model_choices, true_choices, tasks = [], [], []
    num_tasks = targets.shape[0]
    num_trials = env.max_steps

    # loop over dataset making predictions for next trial using model trained on all previous trials
    for task in range(num_tasks):
        trial = env.max_steps-1 # only evaluate on last trial; not possible to evaluate on first trial as it will only have one class
        # loop over trials
        while trial < num_trials:
            trial_inputs = inputs[task, :trial]
            trial_targets = targets[task, :trial]
            try:
                lr_model = LogisticRegressionModel(trial_inputs, trial_targets)
                svm_model = SVMModel(trial_inputs, trial_targets)
                lr_model_choice = lr_model.predict_proba(inputs[task, trial:trial+1])
                svm_model_choice = svm_model.predict_proba(inputs[task, trial:trial+1])
                true_choice = targets[task, trial:trial+1]
                baseline_model_choices.append(torch.tensor([lr_model_choice, svm_model_choice]))
                true_choices.append(true_choice)
                tasks.append(task)
            except:
                print('error')
            trial += 1
    
    # meta-learned model predictions
    _, metal_choice, metal_true_choices = evaluate(env_name=env_name, model_path=model_path, mode=mode, policy='greedy', return_all=True)
    
    # calculate accuracy
    baseline_model_choices, true_choices = torch.stack(baseline_model_choices).squeeze().argmax(2), torch.stack(true_choices).squeeze()
    ml2 = (metal_choice.argmax(2)[-1]==metal_true_choices[-1]).sum()/metal_choice.shape[1]
    accuracy = [(baseline_model_choices[:, model_id] == true_choices).sum()/num_tasks for model_id in range(2)]
    accuracy.append(ml2)      

    # concatenate all model choices and true choices
    all_model_choices = torch.cat([baseline_model_choices, metal_choice.argmax(2)[-1][tasks].unsqueeze(1)], dim=1)
    all_true_choices = true_choices

    # predictions = model_choices.argmax(2).reshape(-1) if policy=='greedy' else \
    #     model_choices.view(model_choices.shape[0]*model_choices.shape[1], model_choices.shape[2]).multinomial(1).reshape(-1)
    
    if return_all:
        return accuracy, all_model_choices, all_true_choices
    else:    
        return accuracy