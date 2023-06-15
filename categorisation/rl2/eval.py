
import numpy as np
import torch
from envs import CategorisationTask
from rl2.rl2 import RL2
import argparse

# evaluate a model
def evaluate(env_name, model_path, mode='val', policy='greedy'):
    
    # load environment
    env = CategorisationTask(data=env_name, mode=mode)
    # load model
    model = torch.load(model_path)[1]
    with torch.no_grad():
        model.eval()

    inputs, targets, done, info = env.reset()
    hx, cx = model.initial_states(env.batch_size)
    model_choices = []
    true_choices = []
    while not done:
        model_choice, hx, cx = model(inputs.unsqueeze(1).float(), hx, cx)
        true_choice = targets
        model_choices.append(model_choice)
        true_choices.append(true_choice)
        inputs, targets, done, info = env.step() 
    model_choices = torch.stack(model_choices).squeeze()
    true_choices = torch.stack(true_choices)

    predictions = model_choices.argmax(2).reshape(-1) if policy=='greedy' else \
          model_choices.view(model_choices.shape[0]*model_choices.shape[1], model_choices.shape[2]).multinomial(1).reshape(-1)
    accuracy = (true_choices.reshape(-1)==predictions).sum()/(predictions.shape[0])
    print(accuracy)