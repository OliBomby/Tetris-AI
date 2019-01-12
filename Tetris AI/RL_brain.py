"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import heapq

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=1000,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            feature_shape=None,
            state_shape=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        if feature_shape is not None:
            self.feature_shape = feature_shape
        else:
            self.feature_shape = n_features
        self.state_shape = state_shape
        self.n_state = state_shape[0] * state_shape[1]

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        def weight_variable(name, shape):
            return tf.get_variable(name, shape, initializer=w_initializer, collections=c_names)

        def bias_variable(name, shape):
            return tf.get_variable(name, shape, initializer=b_initializer, collections=c_names)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('eval_net', reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0.0, 0.1), tf.constant_initializer((-0.01, 0.01))  # config of layers

            # first layer. conv3-32
            w1 = weight_variable('w1', [3, 3, 1, 32])
            b1 = bias_variable('b1', [32])
            # second layer. conv3-32
            w2 = weight_variable('w2', [3, 3, 32, 32])
            b2 = bias_variable('b2', [32])

            # second layer. conv3-64
            w3 = weight_variable('w3', [3, 3, 32, 64])
            b3 = bias_variable('b3', [64])

            # fourth layer. collapse into 1 row 64 channels
            w4 = weight_variable('w4', [self.state_shape[0], 1, 64, 64])
            b4 = bias_variable('b4', [64])

            # fifth layer. conv3-128
            w5 = weight_variable('w5', [1, 3, 64, 128])
            b5 = bias_variable('b5', [128])

            # sixth layer. conv1-128
            w6 = weight_variable('w6', [1, 1, 128, 128])
            b6 = bias_variable('b6', [128])

            # seventh layer. conv3-128
            w7 = weight_variable('w7', [1, 3, 128, 128])
            b7 = bias_variable('b7', [128])

            # eighth layer. FC - 128
            w8 = weight_variable('w8', [128 * self.state_shape[1], 128])
            b8 = bias_variable('b8', [128])

            # nineth layer. FC - 512
            w9 = weight_variable('w9', [128, 512])
            b9 = bias_variable('b9', [512])

            # output layer. FC - 1
            w10 = weight_variable('w10', [512, 1])
            # b10 = bias_variable('b10', [1])
            b10 = tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32), name='b10', collections=c_names)

            s = tf.reshape(self.s, [-1, self.n_actions, self.n_state])
            input_series = tf.unstack(s, axis=1)

            predictions_series = []
            for current_input in input_series:
                s = tf.reshape(current_input, [-1, self.state_shape[0], self.state_shape[1], 1])

                l1 = tf.nn.relu(conv2d(s, w1) + b1)
                l2 = tf.nn.relu(conv2d(l1, w2) + b2)
                l3 = tf.nn.relu(conv2d(l2, w3) + b3)
                l4 = tf.nn.relu(tf.nn.conv2d(l3, w4, strides=[1, 1, 1, 1], padding='VALID') + b4)
                l5 = tf.nn.relu(conv2d(l4, w5) + b5)
                l6 = tf.nn.relu(conv2d(l5, w6) + b6)
                l7 = tf.nn.relu(conv2d(l6, w7) + b7)
                l7_flat = tf.reshape(l7, [-1, 128 * self.state_shape[1]])
                l8 = tf.nn.relu(tf.matmul(l7_flat, w8) + b8)
                l8_drop = tf.nn.dropout(l8, self.keep_prob)
                l9 = tf.nn.relu(tf.matmul(l8_drop, w9) + b9)
                l9_drop = tf.nn.dropout(l9, self.keep_prob)
                l10 = tf.matmul(l9_drop, w10) + b10
                predictions_series.append(l10)

            self.q_eval = tf.reshape(tf.stack(predictions_series, axis=1), [-1, self.n_actions])

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input

        with tf.variable_scope('target_net', reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. conv3-32
            w1 = weight_variable('w1', [3, 3, 1, 32])
            b1 = bias_variable('b1', [32])
            # second layer. conv3-32
            w2 = weight_variable('w2', [3, 3, 32, 32])
            b2 = bias_variable('b2', [32])

            # second layer. conv3-64
            w3 = weight_variable('w3', [3, 3, 32, 64])
            b3 = bias_variable('b3', [64])

            # fourth layer. collapse into 1 row 64 channels
            w4 = weight_variable('w4', [self.state_shape[0], 1, 64, 64])
            b4 = bias_variable('b4', [64])

            # fifth layer. conv3-128
            w5 = weight_variable('w5', [1, 3, 64, 128])
            b5 = bias_variable('b5', [128])

            # sixth layer. conv1-128
            w6 = weight_variable('w6', [1, 1, 128, 128])
            b6 = bias_variable('b6', [128])

            # seventh layer. conv3-128
            w7 = weight_variable('w7', [1, 3, 128, 128])
            b7 = bias_variable('b7', [128])

            # eighth layer. FC - 128
            w8 = weight_variable('w8', [128 * self.state_shape[1], 128])
            b8 = bias_variable('b8', [128])

            # nineth layer. FC - 512
            w9 = weight_variable('w9', [128, 512])
            b9 = bias_variable('b9', [512])

            # output layer. FC - actions
            w10 = weight_variable('w10', [512, 1])
            # b10 = bias_variable('b10', [1])
            b10 = tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32), name='b10', collections=c_names)

            s = tf.reshape(self.s_, [-1, self.n_actions, self.n_state])
            input_series = tf.unstack(s, axis=1)

            predictions_series = []
            for current_input in input_series:
                s = tf.reshape(current_input, [-1, self.state_shape[0], self.state_shape[1], 1])

                l1 = tf.nn.relu(conv2d(s, w1) + b1)
                l2 = tf.nn.relu(conv2d(l1, w2) + b2)
                l3 = tf.nn.relu(conv2d(l2, w3) + b3)
                l4 = tf.nn.relu(tf.nn.conv2d(l3, w4, strides=[1, 1, 1, 1], padding='VALID') + b4)
                l5 = tf.nn.relu(conv2d(l4, w5) + b5)
                l6 = tf.nn.relu(conv2d(l5, w6) + b6)
                l7 = tf.nn.relu(conv2d(l6, w7) + b7)
                l7_flat = tf.reshape(l7, [-1, 128 * self.state_shape[1]])
                l8 = tf.nn.relu(tf.matmul(l7_flat, w8) + b8)
                l8_drop = tf.nn.dropout(l8, self.keep_prob)
                l9 = tf.nn.relu(tf.matmul(l8_drop, w9) + b9)
                l9_drop = tf.nn.dropout(l9, self.keep_prob)
                l10 = tf.matmul(l9_drop, w10) + b10
                predictions_series.append(l10)

            self.q_next = tf.reshape(tf.stack(predictions_series, axis=1), [-1, self.n_actions])

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.get_reading(observation, kp=1)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def get_reading(self, observation, kp=0.75):
        observation = observation[np.newaxis, :]
        return self.sess.run(self.q_eval, feed_dict={self.s: observation, self.keep_prob: kp})

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('target_params_replaced')
            print("Epsilon: %s" % self.epsilon)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # print("generating possible states")
        # # magic
        # neweststates = []
        # fixedstates = []
        # for f in batch_memory:
        #     neweststates.append(self.env.generateAllStates(f[:self.n_features]))
        #     fixedstates.append(self.env.generateAllStates(f[-self.n_features:]))
        # newest = np.stack(neweststates, axis=0)
        # fixed = np.stack(fixedstates, axis=0)


        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
                self.keep_prob: 0.75
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target,
                                                self.keep_prob: 0.75})
        # self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



