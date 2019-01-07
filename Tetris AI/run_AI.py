import tetrisML
from RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

env = tetrisML.TetrisGame()

MEMORY_SIZE = 500000
ACTION_SPACE = env.num_actions
FEATURES = env.num_features
FEATURESHAPE = env.featureShape

sess = tf.Session()

with tf.variable_scope('Double_DQN'):
    # DQN = DeepQNetwork(
    #     n_actions=ACTION_SPACE, n_features=466, memory_size=MEMORY_SIZE, e_greedy=1.00,
    #     e_greedy_increment=0.0001, reward_decay=0.97, double_q=False, sess=sess, output_graph=False)
    #
    DQN = DeepQNetwork(
        n_actions=ACTION_SPACE, n_features=FEATURES, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.0000009, e_greedy=0.9, reward_decay=0.75, output_graph=False, feature_shape=FEATURESHAPE,
        learning_rate=2E-6)

    # DQN = DeepQNetwork(
    #     n_actions=ACTION_SPACE, n_features=FEATURES, memory_size=MEMORY_SIZE,
    #     e_greedy_increment=0.0001, e_greedy=1, reward_decay=0.75, output_graph=False, feature_shape=FEATURESHAPE,
    #     learning_rate=0.001)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
# saver.restore(DQN.sess, "model.ckpt")


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

        if total_steps > 10000:  # learning
            RL.learn()

        if time.time() - startTime > 86400:  # stop game after 24 hours (24 * 60 * 60 seconds)
            break

        observation = observation_
        total_steps += 1
    return RL


def test(RL, frames=10000):
    test_env = tetrisML.TetrisGame(log=True)
    total_steps = 0
    observation = test_env.reset()
    while True:
        actions_value = RL.get_reading(observation, kp=1)
        action = np.argmax(actions_value)
        if action == 40 and not test_env.canUseHold:  # Prevent it from using hold when not available
            action = np.argsort(actions_value)[-2]

        observation_, reward = test_env.nextFrame(action)

        if total_steps > frames:  # stop game
            break

        observation = observation_
        total_steps += 1

    return np.average(test_env.scores), np.average(test_env.gamelengths)


q_natural = train(DQN)

print("Evaluating agent...")
avg_score, avg_length = test(q_natural, 100000)
print("avg score (official): %s" % avg_score)
print("avg game length (frames): %s" % avg_length)

# q_natural.plot_cost()


