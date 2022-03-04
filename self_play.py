import numpy as np
from kaggle_environments import make
from model.ddqn import DDQN
from util.utils import plot_rolling_win_ratio, make_save_name, get_valid_moves, assign_rewards, softmax
from util.agent_tracking import write_data, get_best_k_in_batch, write_validation, get_all_runs
from model.agents import get_agent_map, ddqn_as_agent
import random
import math
import torch
import torch.optim as optim
import datetime

# SEEDS
np.random.seed(42)
random.seed(42)

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

NROWS = 6
NCOLS = 7


def select_agent(ddqn_params, agent_str, agent=None):
    agents = get_agent_map()
    if type(agent_str) == list:
        agent_str = random.choice(agent_str)
    if not agent:
        if agent_str in agents:
            agent = agents[agent_str]
        else:
            a_model = DDQN(**ddqn_params)
            a_model.load_model(agent_str)
            agent, agent_str = ddqn_as_agent(a_model, flip=True)
    return agent, agent_str

def train_agent(ddqn_params, agent_str, agent_str_list=None, agent=None, helping_agent_str=None, num_games=1000, measure_over=500):
    model = DDQN(**ddqn_params)
    agents = get_agent_map()
    if agent_str_list:
        agent, agent_str = select_agent(ddqn_params, agent_str_list)
    else:
        agent, agent_str = select_agent(ddqn_params, agent_str)

    help_agent = None
    if helping_agent_str:
        help_agent = agents[helping_agent_str]

    env = make("connectx", {"rows": NROWS, "columns": NCOLS, "inarow": 4}, debug=True)
    config = env.configuration
    trainer = env.train([None, agent])
    observations = []
    actions = []
    obs = trainer.reset()
    score = []
    while len(score) < num_games:
        observation = [-1 if o == 2 else o for o in obs['board']]
        action = model.select_action(full_context=obs, config=config, agent=help_agent, train=True)
        actions.append(action)
        observations.append(observation)
        obs, reward, done, info = trainer.step(action)
        if done:
            if len(score) == int(num_games / 2) and len(score):
                print(f"Half Way! {datetime.datetime.now()}")
            observation = [-1 if o == 2 else o for o in obs['board']]
            next_observations = [o for o in observations[1:]] + [observation]
            score.append(reward) if reward else score.append(0)
            rewards = assign_rewards(actions, reward)
            model.memory.push([observations, actions, next_observations, rewards])
            model.learn()
            observations = []
            actions = []

            # Re-instantiate the environment
            env.reset()
            if agent_str_list:
                agent, agent_str = select_agent(ddqn_params, agent_str_list)
            else:
                agent, agent_str = select_agent(ddqn_params, agent_str)
            trainer = env.train([None, agent])
            obs = trainer.reset()

    print(f"W/L/D {score.count(1)}, {score.count(-1)}, {score.count(0)}")
    window_scores = score[-measure_over:].count(1)
    last_scores = (round(window_scores/measure_over, 2),
        round(np.var(score[-measure_over:]), 3))
    if agent_str_list:
        agent_str = str(len(agent_str_list))
    plot_rolling_win_ratio(score, window=measure_over, agent=agent_str, save_name=model.name)
    model.save_model(last_scores, ddqn_params)
    write_data(model.name, agent_str, ddqn_params['gen'], last_scores, score.count(1), num_games, ddqn_params)

def train_agent_multiple(ddqn_params, agent_str_list=None, helping_agent_str=None, num_games=1000, measure_over=500):
    model = DDQN(**ddqn_params)
    agents = get_agent_map()
    agent_dict = {}
    for agent in agent_str_list:
        agent_dict[agent], _ = select_agent(ddqn_params, agent)
    first_agent = agent_str_list[0]
    help_agent = None
    if helping_agent_str:
        help_agent = agents[helping_agent_str]

    env = make("connectx", {"rows": NROWS, "columns": NCOLS, "inarow": 4}, debug=True)
    config = env.configuration
    trainer = env.train([None, agent_dict[first_agent]])
    observations = []
    actions = []
    obs = trainer.reset()
    score = []
    while len(score) < num_games:
        observation = [-1 if o == 2 else o for o in obs['board']]
        action = model.select_action(full_context=obs, config=config, agent=help_agent, train=True)
        actions.append(action)
        observations.append(observation)
        obs, reward, done, info = trainer.step(action)
        if done:
            if len(score) == int(num_games / 2) and len(score):
                print(f"Half Way! {datetime.datetime.now()}")
            observation = [-1 if o == 2 else o for o in obs['board']]
            next_observations = [o for o in observations[1:]] + [observation]
            score.append(reward) if reward else score.append(0)
            rewards = assign_rewards(actions, reward)
            model.memory.push([observations, actions, next_observations, rewards])
            model.learn()
            observations = []
            actions = []

            # Re-instantiate the environment
            env.reset()
            agent = random.choice(agent_str_list)
            trainer = env.train([None, agent_dict[agent]])
            obs = trainer.reset()

    print(f"W/L/D {score.count(1)}, {score.count(-1)}, {score.count(0)}")
    window_scores = score[-measure_over:].count(1)
    last_scores = (round(window_scores/measure_over, 2),
        round(np.var(score[-measure_over:]), 3))
    # if agent_str_list:
    agent_str_write = str(len(agent_str_list))
    agent_str = "_".join(agent_str_list)
    plot_rolling_win_ratio(score, window=measure_over, agent=agent_str_write, save_name=model.name)
    model.save_model(last_scores, ddqn_params)
    write_data(model.name, agent_str, ddqn_params['gen'], last_scores, score.count(1), num_games, ddqn_params)

def play_agent(ddqn_params, name, ngames=1000):
    model = DDQN(**ddqn_params)
    model.load_model(name)
    ddqn_agent = ddqn_as_agent(model)
    scores = []
    for _ in range(ngames):
        env = make("connectx", {"rows": NROWS, "columns": NCOLS, "inarow": 4}, debug=True)
        steps = env.run([ddqn_agent, "random"])
        score = steps[-1][0]['reward']
        scores.append(score)
        env.reset()
        env.configuration.randomSeed = None
    print(scores.count(1)/ngames)
    print(scores)


def validate_agent(ddqn_params, name, extra_val_agents=None, val_only=None, ngames=200):
    if extra_val_agents is None:
        extra_val_agents = []
    do_not_val = ['3_ahead', '4_ahead']
    val_agents = [i for i in get_agent_map().keys() if i not in do_not_val] + extra_val_agents
    agents = get_agent_map()

    if val_only:
        if type(val_only) != list:
            val_only = [val_only]
        val_agents = val_only

    for agent_str in val_agents:
        if agent_str in agents:
            agent = agents[agent_str]
        else:
            a_model = DDQN(**ddqn_params)
            a_model.load_model(agent_str)
            agent, _ = ddqn_as_agent(a_model, flip=True)

        model = DDQN(**ddqn_params)
        model.load_model(name)
        ddqn_agent, _ = ddqn_as_agent(model)
        scores = []
        for _ in range(ngames):
            env = make("connectx", {"rows": NROWS, "columns": NCOLS, "inarow": 4}, debug=True)
            steps = env.run([ddqn_agent, agent])
            score = steps[-1][0]['reward']
            scores.append(score)
            env.reset()
        print(scores)
        win_ratio = scores.count(1)/ngames
        print(f"{name} vs {agent_str}: {win_ratio}")
        write_validation(name, agent_str, win_ratio)


def make_fresh_agent_batch(batch_name, N, against="random", helper=None):
    for i in range(N):
        name = "{}{:0>4}_gen0".format(batch_name, i)
        print(f"Training: {name}")
        hidden_layers_count = random.randint(1, 5)
        hidden_layers = sorted([random.randint(NROWS*NCOLS + 1, 1024) for _ in range(hidden_layers_count)])
        layers = tuple([NROWS * NCOLS] + hidden_layers + [NCOLS])
        memory_size = random.choice([5000, 10000])
        random_ddqn_params = {
            "layers": layers,
            "memory_size": memory_size,
            "memory_cutoff": int(memory_size / 5),
            "target_update": random.randint(100, 500),
            "epsilon": {"start": 0.99, "end": 0.05, "decay": 5000, "burn_in": 0, "agent_train": 0},
            "batch_size": random.randint(32, 256),
            "gamma": random.randrange(65, 75, 1)/100,  # Between 0.65 and 0.75
            "name": name,
            "gen": 0
        }
        train_agent(random_ddqn_params, against, helping_agent_str=helper, num_games=1500)


def get_best_k_in_all_previous_batches(old_batch_name, curr_gen, K):
    res = []
    scores = []
    gen = curr_gen - 1
    while gen >= 0:
        res += get_best_k_in_batch(old_batch_name, gen, K)[0]
        scores += get_best_k_in_batch(old_batch_name, gen, K)[1]
        gen -= 1
    return res, scores

def make_next_gen(batch_name, old_batch_name, N, K, gen=1):
    assert gen > 0, "Previous Generation Index invalid."
    # old_agents, scores = get_best_k_in_batch(old_batch_name, gen-1, K)
    # agent_probs = softmax(np.array(scores))
    old_agents, scores = get_best_k_in_all_previous_batches(old_batch_name, gen, K)
    old_agents = old_agents + ['negamax', 'random']
    L = len(old_agents)
    for i in range(N):
        # agent = np.random.choice(old_agents, p=agent_probs)
        name = "{}{:0>4}_gen{}".format(batch_name, i, gen)
        print(f"Training: {name}")
        hidden_layers_count = random.randint(1, 5)
        hidden_layers = sorted([random.randint(NROWS*NCOLS + 1, 1024) for _ in range(hidden_layers_count)])
        layers = tuple([NROWS * NCOLS] + hidden_layers + [NCOLS])
        memory_size = random.choice([5000, 10000])
        random_ddqn_params = {
            "layers": layers,
            "memory_size": memory_size,
            "memory_cutoff": int(memory_size / 5),
            "target_update": random.randint(100, 500),
            "epsilon": {"start": 0.95, "end": 0.05, "decay": 1000*L,"burn_in": 0, "agent_train": 0,},
            "batch_size": random.randint(32, 256),
            "gamma": random.randrange(65, 75, 1) / 100,  # Between 0.65 and 0.75
            "name": name,
            "gen": gen
        }
        train_agent_multiple(random_ddqn_params, agent_str_list=old_agents, num_games=1000 + 300*L)

def play_standard(agent1_str, agent2_str, N):
    agent_1 = get_agent_map()[agent1_str]
    agent_2 = get_agent_map()[agent2_str]
    scores = []
    for i in range(N):
        if i == int(N/2):
            print("Halfway")
        env = make("connectx", {"rows": NROWS, "columns": NCOLS, "inarow": 4}, debug=True)
        steps = env.run([agent_1, agent_2])
        score = steps[-1][0]['reward']
        scores.append(score)
        env.reset()
    print(scores)
    print(scores.count(1)/N)
    print(f"W/L/D {scores.count(1)}, {scores.count(-1)}, {scores.count(0)}")

if __name__ == '__main__':
    hidden_layers_count = random.randint(1, 5)
    hidden_layers = sorted([random.randint(NROWS * NCOLS + 1, 1024) for _ in range(hidden_layers_count)])
    layers = tuple([NROWS * NCOLS] + hidden_layers + [NCOLS])
    # memory_size = random.choice([5000, 10000])
    for i in range(2):
        random_ddqn_params = {
            "layers": layers,
            "memory_size": 100000,
            "memory_cutoff": int(100000 / 5),
            "target_update": 100,
            "epsilon": {"start": 1.0, "end": 0.05, "decay": 50000, "burn_in": 100000, "agent_train": 1},
            "batch_size": random.randint(32, 256),
            "gamma": 1 + (i + 1)/5,  # Between 0.65 and 0.75
            "name": f"1_ahead_gamma_{i}",
            "gen": 0
        }
        against = "1_ahead"
        helper = "1_ahead"
        train_agent(random_ddqn_params, against, helping_agent_str=helper, num_games=20000)

    # make_fresh_agent_batch("build", 20, against="negamax")
    # for i in range(15,25):
    #     make_next_gen("build", "build", 5, 2, i)
    # make_fresh_agent_batch("test3", 10, against="1_ahead")
    # for i in range(1, 4):
    #     make_next_gen("test3", "test3", 10, 3, i)
    # make_fresh_agent_batch("test2", 10, against="2_ahead")
    # for i in range(1, 4):
    #     make_next_gen("test3", "test3", 10, 3, i)
    # names, scores = get_best_k_in_batch("build", [i for i in range(5,25)], 10)
    # names.append("build0000_gen24")
    # # names = get_all_runs(gen=24)
    # print(names, scores)
    # names = ['1_ahead_gamma_3']
    # for name in names:
    #     print(name)
    #     hidden_layers_count = random.randint(1, 5)
    #     hidden_layers = sorted([random.randint(NROWS * NCOLS + 1, 1024) for _ in range(hidden_layers_count)])
    #     layers = tuple([NROWS * NCOLS] + hidden_layers + [NCOLS])
    #     memory_size = random.choice([5000, 10000])
    #     dummy = {
    #             "layers": layers,
    #             "memory_size": memory_size,
    #             "memory_cutoff": int(memory_size / 5),
    #             "target_update": random.randint(100, 500),
    #             "epsilon": {"start": 0.95, "end": 0.05, "decay": 1000, "agent_train": 0, "burn_in": 0},
    #             "batch_size": random.randint(32, 256),
    #             "gamma": random.randrange(65, 75, 1) / 100,  # Between 0.65 and 0.75
    #             "name": name,
    #             "gen": 0
    #         }
    #     val_only = ['1_ahead', "random", "negamax"] + [i for i in names if i != name]
    #     validate_agent(dummy, name, val_only=val_only)
