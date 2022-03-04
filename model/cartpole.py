import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity, cutoff):
        self.capacity = capacity
        self.cutoff = cutoff
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
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
    def __init__(self, layers, memory_size, memory_cutoff, optimizer, epsilon, lr=0.001, batch_size=64, gamma=0.9, target_update=1000):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.target_net = Network(layers).to(self.device)
        self.policy_net = copy.deepcopy(self.target_net).to(self.device)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.memory = ReplayMemory(memory_size, memory_cutoff)
        self.optimizer = optimizer(self.policy_net.parameters(), lr=lr)
        self.epsilon_start = epsilon['start']
        self.epsilon_end = epsilon['end']
        self.epsilon_decay = epsilon['decay']
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, observation, train=True):
        sample = random.random()
        print(observation)
        model = self.policy_net
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if train:
            if sample > eps_threshold:
                # return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
                return model(FloatTensor(observation)).data.max(1)[1].view(1, 1)
            else:
                return LongTensor([[random.randrange(2)]])
        else:
            return model(FloatTensor(observation)).data.max(1)[1].view(1, 1)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))
        print(batch_action.shape)
        batch_reward = Variable(torch.cat(batch_reward))
        batch_next_state = Variable(torch.cat(batch_next_state))

        # current Q values are estimated by NN for all actions
        current_q_values = self.policy_net(batch_state).gather(1, batch_action)
        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.target_net(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        # loss is measured from error between current and newly expected Q values
        # print(current_q_values.shape, expected_q_values.shape)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.reshape(-1, 1))

        # backpropagation of loss to NN
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            print("synced")
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)

    def run_episode(self, episode, env):
        state = env.reset()
        steps = 0
        while True:
            # env.render()
            action = self.select_action(FloatTensor([state]))
            next_state, reward, done, _ = env.step(action.item())
            # negative reward when attempt ends
            if done:
                if steps < 30:
                    reward -= 10
                else:
                    reward = -1
            if steps > 100:
                reward += 1
            if steps > 200:
                reward += 1
            if steps > 300:
                reward += 1

            self.memory.push((FloatTensor([state]),
                         action,  # action is already a tensor
                         FloatTensor([next_state]),
                         FloatTensor([reward])))

            self.learn()

            state = next_state
            steps += 1

            if done or steps >= 1000:
                self.episode_durations.append(steps)
                self.plot_durations()
                print("[Episode {:>5}]  steps: {:>5}".format(episode, steps))
                break
        return False

env = gym.make('CartPole-v0').unwrapped

ddqn_params = {
    "layers": [4, 256, 256, 256, 2],
    "memory_size": 50000,
    "memory_cutoff": 5000,
    "target_update": 1000,
    "optimizer": optim.Adam,
    "epsilon": {"start": 0.95, "end": 0.05, "decay": 500},
    "batch_size": 32,
    "gamma": 0.95,
}

model = DDQN(**ddqn_params)

plt.ion()
for e in range(1000):
    complete = model.run_episode(e, env)

    if complete:
        print('complete...!')
        break
plt.ioff()