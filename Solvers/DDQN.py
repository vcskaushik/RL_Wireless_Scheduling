# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import os
import random
from collections import deque
from keras.layers import Dense, Input, Conv1D, Flatten, Concatenate, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import huber_loss
import numpy as np
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


class DDQN(AbstractSolver):
    def __init__(self,env,options):
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.memory = deque(maxlen=options.replay_memory_size)

    def _build_model(self):
        layers = self.options.layers

        channel_state = Input(shape=(self.env.nT,2))
        power_buffer_state = Input(shape=(3,))

        lrelu = LeakyReLU(alpha=0.5)
        z1 = Conv1D(64,25,strides=2,padding='valid',activation=lrelu)(channel_state)
        z1 = Conv1D(32,3,strides=2,padding='valid',activation=lrelu)(z1)
        z1 = Conv1D(16,3,padding='valid',activation=lrelu)(z1)
        z1 = Conv1D(8,3,padding='valid',activation=lrelu)(z1)

        z1 = Flatten()(z1)
        z2 = Concatenate()([z1,power_buffer_state])
        z2 = Dense(50, activation=lrelu)(z2)
        q = Dense(18, activation='linear')(z2)

        model = Model(inputs=[channel_state,power_buffer_state], outputs=[q])
        model.compile(optimizer=Adam(lr=self.options.alpha), loss=huber_loss)


        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def epsilon_greedy(self, state):
        """
        Apply an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            An epsilon greedy action (as an int) for 'state'

        Use:
            self.env.action_space.n: number of avilable actions
            q_values = self.model.predict([[state]])[0]: Predicted Q values for
                'state' as a vector. One value per action.
            np.argmax(q_values): returns the action coresponding to the highest
                q value
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        nA = self.env.action_space.n
        action_probs = []
        q_values = self.model.predict(state)[0]
        q_len = len(q_values)
        q_ind = range(0,q_len,1)
        if np.random.random()>self.options.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.choice(q_ind)
        return(int(action))

    def replay(self):
        """
        TD learning for q values on past transitions

        Use:
            np.random.choice(len(probs), probs): Randomly select an element
                from probs (a list) based on the probability distribution in probs.
            self.model.predict([[state]])[0]: predicted q values as an array with entry
                per action
        """
        if len(self.memory) > self.options.batch_size:
            minibatch = random.sample(self.memory, self.options.batch_size)
            states = []
            target_q = []
            for state, action, reward, next_state, done in minibatch:
                states.append(state)
                ################################
                #   YOUR IMPLEMENTATION HERE   #
                #  Compute the target value    #
                ################################
        if len(self.memory) > self.options.batch_size:
            minibatch = random.sample(self.memory, self.options.batch_size)
            channel_states = []
            power_buffer_states = []
            target_q = []
            for state, action, reward, next_state, done in minibatch:
                channel_states.append(state[0])
                power_buffer_states.append(state[1])
                ################################
                #   YOUR IMPLEMENTATION HERE   #
                #  Compute the target value    #
                ################################
                q_val = self.model.predict(state)[0]
                action2 = np.argmax(q_val)
                if done:
                    q = reward
                else:
                    q = reward + (self.options.gamma* self.target_model.predict(next_state)[0][action2])
                q_val[action] = q
                target_q.append(q_val)
            channel_states = np.concatenate(channel_states,axis=0)
            power_buffer_states = np.concatenate(power_buffer_states,axis=0)
            states = [channel_states,power_buffer_states]
            target_q = np.array(target_q)
            self.model.fit(states, target_q, epochs=1, verbose=0)



    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Use:
            self.epsilon_greedy(state): return an epsilon greedy action
            self.step(action): advance one step in the environment
            self.memorize(state, action, reward, next_state, done): store the transition in the reply buffer
            self.update_target_model(): copy weights from model to target_model
            self.replay(): TD learning for q values on past transitions
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps
        """

        # Reset the environment
        state = self.env.reset()
        i = 1

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        for step_no in range(self.options.steps):
            # print("hello from while")
            action = self.epsilon_greedy(state)
            new_state, reward, done, _ = self.step(action)
            self.memorize(state, action, reward, new_state, done)
            self.replay()
            i = i + 1
            if step_no % self.options.update_target_estimator_every == 0:
                self.update_target_model()
            if done:
                break
            state = new_state


    def __str__(self):
        return "DDQN"

    def plot(self,stats):
        plotting.plot_episode_stats(stats,plt_name="DDQN2")

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a greedy
            action
        """

        def policy_fn(state):
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

        return policy_fn
