import gym
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from envs import CategorisationTask
from rl2.rl2 import RL2
import argparse
from tqdm import tqdm

def run(env_name, num_episodes, print_every, save_every, num_hidden, save_dir, device, lr, batch_size=32):

    #writer = SummaryWriter('runs/' + save_dir)
    env = CategorisationTask(data=env_name, batch_size=batch_size)
    model = RL2(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden, num_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = [] # keep track of losses
    for t in tqdm(range(int(num_episodes))):
        inputs, targets, done, info = env.reset()
        hx, cx = model.initial_states(batch_size)
        model_choices = []
        true_choices = []
        while not done:
            model_choice, hx, cx = model(inputs.unsqueeze(1).float(), hx, cx)
            true_choice = targets
            model_choices.append(model_choice)
            true_choices.append(true_choice)
            inputs, targets, done, info = env.step() 
        
        # convert to tensors
        model_choices = torch.stack(model_choices)
        true_choices = torch.stack(true_choices)
        # reshape to (batch_size * num_steps, num_choices)
        model_choices = model_choices.view(-1, model_choices.size(-1)).double()
        true_choices = true_choices.view(-1).long()

        # gradient step
        loss = model.compute_loss(model_choices, true_choices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # if (not t % print_every):
        #     writer.add_scalar('Loss', loss, t)

        # if (not t % save_every):
        #     torch.save([c, t, model], save_dir)
    return losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--num-episodes', type=int, default=1e6, help='number of trajectories for training')
    parser.add_argument('--print-every', type=int, default=100, help='how often to print')
    parser.add_argument('--save-every', type=int, default=100, help='how often to save')
    parser.add_argument('--runs', type=int, default=1, help='total number of runs')
    parser.add_argument('--first-run-id', type=int, default=0, help='id of the first run')

    parser.add_argument('--num_hidden', type=int, default=128, help='number of hidden units')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--env-name', default='gershman2018deconstructing-v0', help='name of the environment')
    parser.add_argument('--save-dir', default='trained_models/', help='directory to save models')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.env_name = '/raven/u/ajagadish/vanilla-llama/categorisation/data/llama_generated_tasks_params65B_dim3_data8_tasks14500.csv'# if args.env_name is None else args.env_name

    for i in range(args.runs):
         save_dir = args.save_dir + 'env=' + args.env_name + '_run=' + str(args.first_run_id + i) + '.pt'
         run(args.env_name, args.num_episodes, args.print_every, args.save_every, args.num_hidden, save_dir, device, args.lr)