import pandas as pd
import os
import datetime
import json

CSV_DATAFILE = "./data/agent_history.csv"
VAL_DATAFILE = "./data/agent_validation.csv"
JSON_DATAFILE = "./data/agent_params.json"


def write_training_to_csv(train_agent, target_agent, gen, last_scores, score_total, iterations):
    columns = ['Train', 'Against', 'batch_name', 'gen',
                'Last_Score_Mean', 'Last_Score_Var', 'Total_Score',
               'iters', 'Date']
    if os.path.exists(CSV_DATAFILE):
        df = pd.read_csv(CSV_DATAFILE)
    else:
        df = pd.DataFrame(columns=columns)
    curr_dt = datetime.datetime.now()
    batch_name = train_agent[:train_agent.find('0')]
    df = pd.concat([df,
        pd.DataFrame([[train_agent, target_agent, batch_name, gen,
                       last_scores[0], last_scores[1], score_total,
                       iterations, curr_dt]],
                     columns=columns)
                   ])
    df.to_csv(CSV_DATAFILE, index=False)
    return df

def get_best_k_in_batch(batch, gen, k):
    if type(batch) not in (list, tuple):
        batch = [batch]
    if type(gen) not in (list, tuple):
        gen = [gen]
    df = pd.read_csv(CSV_DATAFILE)
    df = df[df['batch_name'].isin(batch) & df['gen'].isin(gen)].copy()
    df_sorted = df.sort_values(by=['Last_Score_Mean'], ascending=False)
    best = df_sorted['Train'].iloc[:k]
    scores = df_sorted['Last_Score_Mean'].iloc[:k]
    return list(best), list(scores)


def write_params_to_json(ddqn_params):
    if os.path.exists(JSON_DATAFILE):
        js = json.load(open(JSON_DATAFILE, 'r'))
    else:
        js = {"models": {}}
    js['models'][ddqn_params['name']] = ddqn_params
    json.dump(js, open(JSON_DATAFILE, 'w'), indent=4)


def write_data(train_agent, target_agent, gen, last_scores, score_total, iterations, ddqn_params):
    write_training_to_csv(train_agent, target_agent, gen, last_scores, score_total, iterations)
    write_params_to_json(ddqn_params)


def write_validation(train_agent, against, against_score):
    columns = ['Train', 'Against', 'Score', 'Date']
    if os.path.exists(VAL_DATAFILE):
        df = pd.read_csv(VAL_DATAFILE)
    else:
        df = pd.DataFrame(columns=columns)
    curr_dt = datetime.datetime.now()
    df = pd.concat([df,
                    pd.DataFrame([[train_agent, against, against_score, curr_dt]],
                                 columns=columns)
                    ])
    df.to_csv(VAL_DATAFILE, index=False)
    return df

def get_all_runs(batch = None, gen=None):
    df = pd.read_csv(CSV_DATAFILE)
    df = df[df['batch'] == batch].copy() if batch else df
    df = df[df['gen'] == gen].copy() if gen else df
    return list(df['Train'])
