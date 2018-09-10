import numpy as np

WIDTH = 10
HEIGHT = 30

PLAYING_FIELD_DIMENSIONS = (WIDTH, HEIGHT)


def detect_full_rows(field):
    return [i for i, r in enumerate(field) if np.mean(r) == 1]


def get_next_game_state(state, game_input=None):
    if game_input is None:
        game_input = [0 for x in range(5)]
    field = state[0]

    index_full_rows = detect_full_rows(field)
    num_full_rows = len(index_full_rows)
    # move piece down
    # respond to input
    # check for completed rows
    # do stuff
    # cycle pieces or something
    # generate new state


def draw_game_state(state):
    # display the game state
    pass


def game_loop():
    # artificial loop for human play

    field = np.zeros(PLAYING_FIELD_DIMENSIONS, dtype=np.float32)



    state = (field)

    while True:
        # check inputs
        # get_next_game_state
        # draw_game_state
        pass

