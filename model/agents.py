import random
import numpy as np

def play(board, column, mark, config):
    EMPTY = 0
    columns = config.columns
    rows = config.rows
    row = max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
    board[column + (row * columns)] = mark


def is_win(board, column, mark, config, has_played=True):
    EMPTY = 0
    columns = config.columns
    rows = config.rows
    inarow = config.inarow - 1
    row = (
        min([r for r in range(rows) if board[column + (r * columns)] == mark])
        if has_played
        else max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
    )

def negamax_agent(obs, config):
    columns = config.columns
    rows = config.rows
    size = rows * columns
    from random import choice  # connect the library for working with random numbers

    # Due to compute/time constraints the tree depth must be limited.
    max_depth = 4
    EMPTY = 0

    def negamax(board, mark, depth):
        moves = sum(1 if cell != EMPTY else 0 for cell in board)

        # Tie Game
        if moves == size:
            return (0, None)

        # Can win next.
        for column in range(columns):
            if board[column] == EMPTY and is_win(board, column, mark, config, False):
                return ((size + 1 - moves) / 2, column)

        # Recursively check all columns.
        best_score = -size
        best_column = None
        for column in range(columns):
            if board[column] == EMPTY:
                # Max depth reached. Score based on cell proximity for a clustering effect.
                if depth <= 0:
                    row = max(
                        [
                            r
                            for r in range(rows)
                            if board[column + (r * columns)] == EMPTY
                        ]
                    )
                    score = (size + 1 - moves) / 2
                    if column > 0 and board[row * columns + column - 1] == mark:
                        score += 1
                    if (
                        column < columns - 1
                        and board[row * columns + column + 1] == mark
                    ):
                        score += 1
                    if row > 0 and board[(row - 1) * columns + column] == mark:
                        score += 1
                    if row < rows - 2 and board[(row + 1) * columns + column] == mark:
                        score += 1
                else:
                    next_board = board[:]
                    play(next_board, column, mark, config)
                    (score, _) = negamax(next_board,
                                         1 if mark == 2 else 2, depth - 1)
                    score = score * -1
                if score > best_score or (score == best_score and choice([True, False])):
                    best_score = score
                    best_column = column

        return (best_score, best_column)

    _, column = negamax(obs.board[:], obs.mark, max_depth)
    if column == None:
        column = choice([c for c in range(columns) if obs.board[c] == EMPTY])
    return column

'''
Helper Functions:
- score_move_a: calculates score if agent drops piece in selected column
- score_move_b: calculates score if opponent drops piece in selected column
- drop_piece: return grid status after player drops a piece
- get_heuristic: calculates value of heuristic for grid
- get_heuristic_optimised: calculates value of heuristic optimised
- check_window: checks if window satisfies heuristic conditions
- count_windows: counts number of windows satisfying specified heuristic conditions
- count_windows_optimised: counts number of windows satisfying specified heuristic optimised conditions
'''

# Calculates score if agent drops piece in selected column
def score_move_a(grid, col, mark, config, start_score, n_steps):
    next_grid, pos = drop_piece(grid, col, mark, config)
    row, col = pos
    score = get_heuristic_optimised(grid,next_grid,mark,config, row, col,start_score)
    valid_moves = [col for col in range (config.columns) if next_grid[0][col]==0]
    '''Since we have just dropped our piece there is only the possibility of us getting 4 in a row and not the opponent.
    Thus score can only be +infinity'''
    scores = []
    if len(valid_moves)==0 or n_steps ==0 or score == float("inf"):
        return score
    else :
        for col in valid_moves:
            current = score_move_b(next_grid,col,mark,config,score,n_steps-1)
            scores.append(current)
        score = min(scores)
    return score

# calculates score if opponent drops piece in selected column
def score_move_b(grid, col, mark, config, start_score, n_steps):
    next_grid, pos = drop_piece(grid,col,(mark%2)+1,config)
    row, col = pos
    score = get_heuristic_optimised(grid,next_grid,mark,config, row, col,start_score)
    valid_moves = [col for col in range (config.columns) if next_grid[0][col]==0]
    '''
    Since we have just dropped opponent piece there is only the possibility of opponent getting 4 in a row and not us.
    Thus score can only be -infinity.
    '''
    scores = []
    if len(valid_moves)==0 or n_steps ==0 or score == float ("-inf"):
        return score
    else :
        for col in valid_moves:
            current = score_move_a (next_grid,col,mark,config,score,n_steps-1)
            scores.append(current)
        score = max(scores)
    return score

# Gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):  #
    next_grid = grid.copy()  # make a copy of the location of the chips on the playing field for its further transformation
    for row in range(config.rows-1, -1, -1):  # iterate over all rows in the playing field
        if next_grid[row][col] == 0:  # we are not interested in empty cells
            break  # we skip them if we meet such
    next_grid[row][col] = mark # mark the cell in which our chip will fall
    return next_grid,(row,col) # return board at next step

# calculates value of heuristic for grid
def get_heuristic(grid, mark, config):
    score = 0
    num = count_windows(grid,mark,config)
    for i in range(config.inarow):
        #num  = count_windows (grid,i+1,mark,config)
        if (i==(config.inarow-1) and num[i+1] >= 1):
            return float("inf")
        score += (4**(i))*num[i+1]
    num_opp = count_windows (grid,mark%2+1,config)
    for i in range(config.inarow):
        if (i==(config.inarow-1) and num_opp[i+1] >= 1):
            return float ("-inf")
        score-= (2**((2*i)+1))*num_opp[i+1]
    return score

# calculates value of heuristic optimised
def get_heuristic_optimised(grid, next_grid, mark, config, row, col, start_score):
    score = 0
    num1 = count_windows_optimised(grid,mark,config,row,col)
    num2 = count_windows_optimised(next_grid,mark,config,row,col)
    for i in range(config.inarow):
        if (i==(config.inarow-1) and (num2[i+1]-num1[i+1]) >= 1):
            return float("inf")
        score += (4**(i))*(num2[i+1]-num1[i+1])
    num1_opp = count_windows_optimised(grid,mark%2+1,config,row,col)
    num2_opp = count_windows_optimised(next_grid,mark%2+1,config,row,col)
    for i in range(config.inarow):
        if (i==(config.inarow-1) and num2_opp[i+1]-num1_opp[i+1]  >= 1):
            return float ("-inf")
        score-= (2**((2*i)+1))*(num2_opp[i+1]-num1_opp[i+1])
    score+= start_score
    return score

# checks if window satisfies heuristic conditions
def check_window(window, piece, config):
    if window.count((piece%2)+1)==0:
        return window.count(piece)
    else:
        return -1

# counts number of windows satisfying specified heuristic conditions
def count_windows(grid, piece, config):
    num_windows = np.zeros(config.inarow+1)
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    return num_windows

# counts number of windows satisfying specified heuristic optimised conditions
def count_windows_optimised(grid, piece, config, row, col):
    num_windows = np.zeros(config.inarow+1)
    # horizontal
    for acol in range(max(0,col-(config.inarow-1)),min(col+1,(config.columns-(config.inarow-1)))):
        window = list(grid[row, acol:acol+config.inarow])
        type_window = check_window(window, piece, config)
        if type_window != -1:
            num_windows[type_window] += 1
    # vertical
    for arow in range(max(0,row-(config.inarow-1)),min(row+1,(config.rows-(config.inarow-1)))):
        window = list(grid[arow:arow+config.inarow, col])
        type_window = check_window(window, piece, config)
        if type_window != -1:
            num_windows[type_window] += 1
    # positive diagonal
    for arow, acol in zip(range(row-(config.inarow-1),row+1),range(col-(config.inarow-1),col+1)):
        if (arow>=0 and acol>=0 and arow<=(config.rows-config.inarow) and acol<=(config.columns-config.inarow)):
            window = list(grid[range(arow, arow+config.inarow), range(acol, acol+config.inarow)])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    # negative diagonal
    for arow,acol in zip(range(row,row+config.inarow),range(col,col-config.inarow,-1)):
        if (arow >= (config.inarow-1) and acol >=0 and arow <= (config.rows-1) and acol <= (config.columns-config.inarow)):
            window = list(grid[range(arow, arow-config.inarow, -1), range(acol, acol+config.inarow)])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    return num_windows

# main function of our agent
def n_ahead_fast_agent(n):
    def agent(obs, config):
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]
        scores = {}
        start_score = get_heuristic(grid, obs.mark, config)
        for col in valid_moves:
            scores[col] = score_move_a(grid, col, obs.mark, config,start_score, n)
        max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
        return random.choice(max_cols)
    return agent


def ddqn_as_agent(ddqn, flip=False):
    model = ddqn

    def agent(obs, config):
        action = model.select_action(obs, train=False, flip=flip)
        return action

    agent_name = model.name

    return agent, agent_name

def get_agent_map():
    agents = {
        "random": "random",
        "negamax": negamax_agent,
        "1_ahead": n_ahead_fast_agent(1),
        "2_ahead": n_ahead_fast_agent(2),
        "3_ahead": n_ahead_fast_agent(3),
        "4_ahead": n_ahead_fast_agent(4)
    }
    return agents
