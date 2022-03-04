import numpy as np
import streamlit as st
import requests
import json

# GLOBAL CONFIG
IMG_FOLDER = "./frontend/imgs/"
NROWS = 6
NCOLS = 7


# Setup API query structures and perform a cold call at app load.
# This will make the gameplay smoother once the user starts.
host_address = st.secrets['host_address']
ai_url = st.secrets['ai_url']

headers = {
            "accept": "*/*",
            "content-type": "text/plain",
            "host": host_address,
        }

ddqn_url = ai_url
myobj = {"obs": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 1, 1]}}

# Cold call
x = requests.post(ddqn_url,headers=headers, data=json.dumps(myobj))

# Setup state-space.
if 'button_disable' not in st.session_state:
    st.session_state.button_disable = False

if 'connect' not in st.session_state:
    st.session_state.connect = np.zeros(shape=(NROWS, NCOLS))
    st.session_state.connect[NROWS - 1, 3] = 1

# Game logic and win-conditions
def is_win(seq, mark):
    for i in seq:
        if i != mark:
            return False
    return True

def break_into_l_list(x, l=4):
    if len(x) <= 4:
        return x
    else:
        return [break_into_l_list(x[i:i+l]) for i in range(len(x) - l + 1)]

def get_diags(a, l=4):
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1, a.shape[1])]
    diags.extend(a.diagonal(i) for i in range(a.shape[1]-1, -a.shape[0], -1))
    res = [n.tolist() for n in diags if len(n) >= l]
    new_list = []
    for x in res:
        if len(x) == 4:
            new_list.append(x)
        else:
            l = break_into_l_list(x, l=4)
            for i in l:
                new_list.append(i)
    return new_list

def check_win():
    player1_win = (True, "AI Wins")
    player2_win = (True, "Player Wins")
    no_win = (False, "Nobody Yet")

    board = st.session_state.connect
    # Horizontal
    for i in range(NROWS):
        for j in range(NCOLS-4 + 1):
            seq = board[i, j:(j+4)]
            if is_win(seq, 1):
                return player1_win
            if is_win(seq, -1):
                return player2_win
    # Vertical
    for i in range(NROWS-4 + 1):
        for j in range(NCOLS):
            seq = board[i:(i+4), j]
            if is_win(seq, 1):
                return player1_win
            if is_win(seq, -1):
                return player2_win

    # Diagonals
    diagonals = get_diags(board)
    for seq in diagonals:
        if is_win(seq, 1):
            return player1_win
        if is_win(seq, -1):
            return player2_win

    return no_win


def enemy_play():
    curr_board = list(st.session_state.connect.reshape(NROWS*NCOLS))
    payload = {
        "obs":
            {"board": curr_board}
    }
    req = requests.post(ddqn_url, headers=headers, data=json.dumps(payload))
    action = json.loads(req.text)['prediction']
    return action

def win_process():
    st.balloons()
    st.success("You win! Congrats!")

def loss_process():
    st.warning("Failure... the AI beat you!")

def end_game(win_status):
    st.session_state.button_disable = True

    if win_status[1] == "AI Wins":
        loss_process()
    if win_status[1] == "Player Wins":
        win_process()

def update_board(col):
    row = NROWS - 1
    while st.session_state.connect[row, col]:
        row -= 1
        if row < 0:
            break
    st.session_state.connect[row, col] = -1

    win_check = check_win()
    if win_check[0]:
        end_game(win_check)
        return

    enemy_action = enemy_play()
    row = NROWS - 1
    while st.session_state.connect[row, enemy_action]:
        row -= 1
        if row < 0:
            break
    st.session_state.connect[row, enemy_action] = 1

    win_check = check_win()
    if win_check[0]:
        end_game(win_check)
        return

def select_img(mark):
    if mark == -1:
        return f"{IMG_FOLDER}circle_blue.png"
    elif mark == 1:
        return f"{IMG_FOLDER}circle_red.png"
    else:
        return f"{IMG_FOLDER}circle_white.png"


def reset_game():
    st.session_state.button_disable = False
    st.session_state.connect = np.zeros(shape=(NROWS, NCOLS))
    st.session_state.connect[NROWS - 1, 3] = 1


# UI
st.title('Connect Four!')
st.subheader("Can you beat the AI?")

cols = st.columns(NCOLS)
images = [[cols[i].image(select_img(st.session_state.connect[j, i])) for i in range(NCOLS)] for j in range(NROWS)]
buttons = [
    cols[i].button("Play", key=i, disabled=st.session_state.button_disable, on_click=update_board, args=(i,))
    for i in range(NCOLS)
]
st.button("Reset Game", on_click=reset_game)
