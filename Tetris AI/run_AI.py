import tetrisML
from RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = tetrisML.TetrisGame()

MEMORY_SIZE = 1000000
ACTION_SPACE = env.num_actions
FEATURES = env.num_features
FEATURESHAPE = env.featureShape

sess = tf.Session()

with tf.variable_scope('Double_DQN'):
    # DQN = DeepQNetwork(
    #     n_actions=ACTION_SPACE, n_features=466, memory_size=MEMORY_SIZE, e_greedy=1.00,
    #     e_greedy_increment=0.0001, reward_decay=0.97, double_q=False, sess=sess, output_graph=True)

    DQN = DeepQNetwork(
        n_actions=ACTION_SPACE, n_features=FEATURES, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.0000009, e_greedy=0.9, reward_decay=0.75, output_graph=True, feature_shape=FEATURESHAPE,
        learning_rate=2E-6)


sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)
        if action == 40 and not env.canUseHold:  # Prevent it from using hold when not available
            action = np.random.randint(0, 40)

        # f_action = np.zeros(ACTION_SPACE)
        # f_action[action] = 1
        # print(f_action)
        # f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.nextFrame(action)

        reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 10000:  # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 999999999999:  # stop game
            break

        observation = observation_
        total_steps += 1
    return RL


q_natural = train(DQN)

q_natural.plot_cost()
# plt.plot(np.array(q_natural), c='r', label='natural')
# plt.legend(loc='best')
# plt.ylabel('Q eval')
# plt.xlabel('training steps')
# plt.grid()
# plt.show()
