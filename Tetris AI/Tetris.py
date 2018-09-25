import copy

import numpy as np

WIDTH = 10
HEIGHT = 40

PLAYING_FIELD_DIMENSIONS = (WIDTH, HEIGHT)
NUM_UPCOMING = 4
NUM_TETROMINO = 7

TETROMINO = []

class State:
    def __init__(self, fill=True):
        self.field = np.zeros(PLAYING_FIELD_DIMENSIONS, dtype=np.float32)
        self.active = np.zeros(NUM_TETROMINO, dtype=np.float32)
        self.active_pos = [0, 0]
        self.active_rot = 0
        self.hold = np.zeros(NUM_TETROMINO, dtype=np.float32)
        self.upcoming = np.zeros((NUM_UPCOMING, NUM_TETROMINO), dtype=np.float32)

        if fill:
            self.active[np.random.choice(NUM_TETROMINO)] = 1
            self.upcoming = np.eye(NUM_TETROMINO)[np.random.choice(NUM_TETROMINO, NUM_UPCOMING)]


def detect_full_rows(field):
    return [i for i, r in enumerate(field) if np.mean(r) == 1]


def get_next_game_state(state, game_input=None):
    if game_input is None:
        game_input = [0 for x in range(5)]

    state = copy.deepcopy(state)

    index_full_rows = detect_full_rows(state)
    num_full_rows = len(index_full_rows)
    # move piece down
    # respond to input
    # check for completed rows
    # do stuff
    # cycle pieces or something
    # generate new state
    return state


def draw_game_state(state):
    # display the game state
    pass


def check_inputs():
    pass


def game_loop():
    # artificial loop for human play

    state = State()

    while True:
        # check inputs
        game_input = check_inputs()
        # get_next_game_state
        state = get_next_game_state(state, game_input)
        # draw_game_state
        draw_game_state(state)

game_loop()
