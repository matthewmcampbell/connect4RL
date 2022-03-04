import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import copy


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

DATAPATH = "./model/"
def get_valid_moves(observation, ncols):
    return Tensor([[1 if o == 0 else 0 for o in observation[:ncols]]])

class ReplayMemory:
    def __init__(self, capacity, cutoff):
        self.capacity = capacity
        self.cutoff = cutoff
        self.memory = []

    def push(self, transition):
        for i in range(len(transition[0])):
            trans_data = [FloatTensor([transition[j][i]]) for j in range(4)]
            self.memory.append(trans_data)
            if len(self.memory) > self.capacity:
                del self.memory[self.cutoff]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.linears[i](x))
        return x

class DDQN:
    def __init__(self, layers, memory_size, memory_cutoff, epsilon,
                 lr=0.001, batch_size=64, gamma=0.9, target_update=1000, name="no_name",
                 gen=0, ncols=7, nrows=6):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.choices = layers[-1]  # Last layer should be number of choices available
        self.target_net = Network(layers).to(self.device)
        self.policy_net = copy.deepcopy(self.target_net).to(self.device)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.memory = ReplayMemory(memory_size, memory_cutoff)
        self.lr = lr
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.epsilon_start = epsilon['start']
        self.epsilon_end = epsilon['end']
        self.epsilon_decay = epsilon['decay']
        self.epsilon_agent_train = epsilon['agent_train']
        self.burn_in = epsilon['burn_in']
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.steps_done = 0
        self.target_update_steps = 0
        self.episode_durations = []
        self.name = name
        self.gen = gen
        self.ncols = ncols
        self.nrows = nrows


    def save_model(self, avg_score, ddqn_params):
        torch.save({
            'policy_net_dict': self.policy_net.state_dict(),
            'target_net_dict':  self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'avg_score': avg_score,
            'memory': self.memory,
            'ddqn_params': ddqn_params,
        }, f"{DATAPATH}{self.name}.pt")

    def load_model(self, name):
        loaded_data = torch.load(f"{DATAPATH}{name}.pt", map_location=torch.device('cpu'))
        # Reconstruct networks/optimizer from parameterization
        layers = loaded_data['ddqn_params']['layers']

        self.target_net = Network(layers).to(self.device)
        self.policy_net = copy.deepcopy(self.target_net).to(self.device)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Load in previous weights and memory state
        self.policy_net.load_state_dict(loaded_data['policy_net_dict'])
        self.target_net.load_state_dict(loaded_data['target_net_dict'])
        self.optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
        self.memory = loaded_data['memory']
        self.name = name
        print(f"Agent {self.name} Loaded!")

    def select_action(self, full_context=None, config=None, agent=None, train=True, flip=False):
        observation = [-1 if o == 2 else o for o in full_context['board']]
        valid_action = get_valid_moves(observation, self.ncols)
        valid_cols = [i for i in range(self.choices) if valid_action[0][i]]
        flip = -1 if flip else 1
        observation = flip * FloatTensor(observation).reshape(1, -1)
        model = self.policy_net

        def ensure_valid_action():
            action = (model(FloatTensor(observation)) * valid_action).data.max(1)[1].view(1, 1).item()
            if action not in valid_cols:
                # print("this", valid_cols)
                action = random.choice([i for i in range(self.choices) if valid_action[0][i]])
            return action

        if train:
            eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
                -1. * max(self.steps_done - self.burn_in, 0) / self.epsilon_decay)
            sample = random.random()

            self.steps_done += 1
            self.target_update_steps += 1
            if self.steps_done == self.burn_in:
                print("BURNED", self.steps_done)
            if self.steps_done == self.epsilon_decay + self.burn_in:
                print("FULL DECAY", self.steps_done)
            # print(eps_threshold)
            if sample > eps_threshold:
                return ensure_valid_action()
            else:
                if agent:
                    sample = random.random()
                    if sample < self.epsilon_agent_train:
                        action = agent(full_context, config)
                        return action
                choice = random.choice([i for i in range(self.choices) if valid_action[0][i]])
                return choice
        else:
            return ensure_valid_action()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state))
        batch_action = torch.cat(batch_action).to(torch.int64).reshape(-1, 1)
        batch_reward = Variable(torch.cat(batch_reward))
        batch_next_state = Variable(torch.cat(batch_next_state))

        # current Q values are estimated by NN for all actions
        current_q_values = self.policy_net(batch_state).gather(1, batch_action)
        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.target_net(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.reshape(-1, 1))

        # backpropagation of loss to NN
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.target_update_steps > self.target_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_update_steps = 0