import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from envs import SmithsTask


task = SmithsTask(rule='linear')
task.sample_batch()