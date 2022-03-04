"""
Contains general purpose utility functions
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def make_readable_board(L, ncols=7):
    return L.reshape((-1, ncols))


def make_save_name(dic):
    res = ", ".join([f"{key}-{dic[key]}" for key in dic.keys()])
    return res


def get_valid_moves(observation, ncols):
    return Tensor([[1 if o == 0 else 0 for o in observation[:ncols]]])


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def assign_rewards(actions, reward=0):
    if reward is None:
        reward = 0
    return [reward for i in actions]


def plot_rolling_win_ratio(scores: list, window=500, agent=None, save_name=None):
    data = [scores[i: i+window].count(1) / window for i in range(len(scores) - window)]
    x = np.arange(window, len(scores))
    plt.plot(x, data)
    plt.ylim(-1, 1)
    if save_name:
        if not os.path.exists(f"./data/{agent}/"):
            os.mkdir(f"./data/{agent}")
        plt.savefig(f"./data/{agent}/{save_name}.png")
        plt.clf()
    else:
        plt.show()
