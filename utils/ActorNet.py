import os
from distributed import Actor
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorNet(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(ActorNet, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_path = os.path.join(self.chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.ln1 = nn.LayerNorm(self.fc1_dims)
        self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        fan1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-fan1, fan1)
        self.fc1.bias.data.uniform_(-fan1, fan1)
        
        fan2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-fan2, fan2)
        self.fc2.bias.data.uniform_(-fan2, fan2)

        f3 = 3e-3
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        out = torch.tanh(self.mu(x))
        return out

    def save_checkpoint(self):
        print(' Saving model checkpoint ')
        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        print(' Loading model checkpoint ')
        self.load_state_dict(torch.load(self.checkpoint_path))