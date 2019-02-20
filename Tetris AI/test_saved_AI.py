import tetrisML
from RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from os import path


def train(RL):
    startTime = time.time()

    total_steps = 0
    observation = env.reset()
    while True:
        action = RL.choose_action(observation)
        if action == 40 and not env.canUseHold:  # Prevent it from using hold when not available
            action = np.random.randint(0, 40)

        observation_, reward = env.nextFrame(action)

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 5000:  # learning
            RL.learn()

        if time.time() - startTime > 86400:  # stop game after 24 hours (24 * 60 * 60 seconds)
            break

        observation = observation_
        total_steps += 1
    return RL


def testAgent(RL, test, frames=2000):
    test_env = tetrisML.TetrisGame("Testing " + test[0], test[1], test[2], test[3], log=True)
    total_steps = 0
    observation = test_env.reset()
    while True:
        actions_value = RL.get_reading(observation, kp=1)[0]
        action = np.argmax(actions_value)
        if action == 40 and not test_env.canUseHold:  # Prevent it from using hold when not available
            action = np.argsort(actions_value)[-2]

        observation_, reward = test_env.nextFrame(action)

        if total_steps > frames:  # stop game
            break

        observation = observation_
        total_steps += 1

    return np.average(test_env.scores), np.average(test_env.gamelengths), np.average(test_env.scoreChanges), np.average(
        test_env.heuristicChanges)


def testRandom(test, frames=2000):
    test_env = tetrisML.TetrisGame("Testing " + test[0], test[1], test[2], test[3], log=True)
    total_steps = 0
    observation = test_env.reset()
    while True:
        action = np.random.randint(0, 41)

        observation_, reward = test_env.nextFrame(action)

        if total_steps > frames:  # stop game
            break

        observation = observation_
        total_steps += 1

    return np.average(test_env.scores), np.average(test_env.gamelengths), np.average(test_env.scoreChanges), np.average(
        test_env.heuristicChanges)


tests = [
    ["Controle", True, True, 6],
    ["Onzichtbaar veld", False, True, 6],
    ["Onzichtbare hold", True, False, 6],
    ["Een next piece zichtbaar", True, True, 1],
    ["Geen next piece zichtbaar", True, True, 0]
]

tests = [
    ["Controle", True, True, 6]
]

folder = path.dirname(__file__)


for test in tests:
    env = tetrisML.TetrisGame("Training " + test[0], test[1], test[2], test[3])

    MEMORY_SIZE = 100
    ACTION_SPACE = env.num_actions
    FEATURES = env.num_features
    FEATURESHAPE = env.featureShape
    STATESHAPE = env.stateShape

    sess = tf.Session()

    with tf.variable_scope('Double_DQN'):
        DQN = DeepQNetwork(
            n_actions=ACTION_SPACE, n_features=FEATURES, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.0000045, e_greedy=0.9, reward_decay=0.75, output_graph=False,
            feature_shape=FEATURESHAPE,
            state_shape=STATESHAPE, learning_rate=2E-6)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(DQN.sess, path.join(folder, test[0] + ".ckpt"))

    print("Evaluating agent...")
    avg_score, avg_length, scoreChange, heuristicChange = testAgent(DQN, test, frames=5000)
    print("avg score (official): %s" % avg_score)
    print("avg game length (frames): %s" % avg_length)
    print("avg score change per frame: %s" % scoreChange)
    print("avg heuristic score change per frame: %s" % heuristicChange)

    del DQN
