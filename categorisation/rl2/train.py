import gym
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from envs import CategorisationTask
from model import RL2
import argparse
from tqdm import tqdm
from evaluate import evaluate

def run(env_name, num_episodes, print_every, save_every, num_hidden, save_dir, device, lr, batch_size=32):

    writer = SummaryWriter('runs/' + save_dir)
    env = CategorisationTask(data=env_name, batch_size=batch_size, device=device).to(device)
    model = RL2(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden, num_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = [] # keep track of losses
    accuracy = [] # keep track of accuracies
    for t in tqdm(range(int(num_episodes))):
        inputs, targets, prev_targets, done, info = env.reset()
        hx, cx = model.initial_states(batch_size)
        model_choices = []
        true_choices = []
        
        while not done:
            inputs = model.make_inputs(inputs, prev_targets) 
            model_choice, hx, cx = model(inputs.float(), hx, cx)
            true_choice = targets.detach().clone()
            model_choices.append(model_choice)
            true_choices.append(true_choice)
            inputs, targets, prev_targets, done, info = env.step() 
        
        # convert to tensors
        model_choices = torch.stack(model_choices)
        true_choices = torch.stack(true_choices)
        # reshape to (batch_size * num_steps, num_choices)
        model_choices = model_choices.view(-1, model_choices.size(-1)).double()
        true_choices = true_choices.view(-1)

        # gradient step
        loss = model.compute_loss(model_choices, true_choices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        losses.append(loss.item())
        
        if (not t % print_every):
            writer.add_scalar('Loss', loss, t)
            

        if (not t % save_every):
            torch.save([t, model], save_dir)
            acc = evaluate(env_name=env_name, model_path=save_dir, mode='val', policy='greedy')
            accuracy.append(acc)
            writer.add_scalar('Val. Acc.', acc, t)
        
    return losses, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='meta-learning for categorisation')
    parser.add_argument('--num-episodes', type=int, default=1e6, help='number of trajectories for training')
    parser.add_argument('--print-every', type=int, default=100, help='how often to print')
    parser.add_argument('--save-every', type=int, default=100, help='how often to save')
    parser.add_argument('--runs', type=int, default=1, help='total number of runs')
    parser.add_argument('--first-run-id', type=int, default=0, help='id of the first run')

    parser.add_argument('--num_hidden', type=int, default=128, help='number of hidden units')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--env-name', default='llama_generated_tasks_params65B_dim3_data8_tasks14500', help='name of the environment')
    parser.add_argument('--env-dir', default='raven/u/ajagadish/vanilla-llama/categorisation/data', help='name of the environment')
    parser.add_argument('--save-dir', default='trained_models/', help='directory to save models')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cpu" #torch.device("cuda" if use_cuda else "cpu")
    env_name = f'/{args.env_dir}/{args.env_name}.csv'# if args.env_name is None else args.env_name

    for i in range(args.runs):
         save_dir = f'{args.save_dir}env={args.env_name}_num_episodes{str(args.num_episodes)}_num_hidden={str(args.num_hidden)}_lr{str(args.lr)}_run={str(args.first_run_id + i)}.pt'
         run(env_name, args.num_episodes, args.print_every, args.save_every, args.num_hidden, save_dir, device, args.lr)