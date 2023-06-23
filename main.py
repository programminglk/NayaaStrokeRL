import sys
import numpy as np
import math
import random

import gymnasium as gym
import gym_game


def find_closest_key(key):   
    index_1 = key[1]
    index_2 = key[2]

    # get the multiple of 10 that is lower than index_1
    index_1 = index_1 - (index_1 % 5)

    # get the multiple of 10 that is lower than index_2
    index_2 = index_2 - (index_2 % 5)

    return tuple([key[0], index_1, index_2])


def simulate():
    global epsilon, epsilon_decay
    for episode in range(MAX_EPISODES):

        # Init environment
        state, info = env.reset()
        stata = tuple(state)
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):

            # In the beginning, do random action to learn
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[find_closest_key(state)])

            print("action: ", action)

            # Do action and get result
            next_state, reward, done, truncated, info = env.step(action)
            next_state = tuple(next_state)
            total_reward += reward

            print("state: ", state)
            print("next state: ", next_state)

            # Get correspond q value from state, action pair
            q_value = q_table[find_closest_key(tuple(state))][action]
            best_q = max(q_table[find_closest_key(next_state)].values())

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            print("q table state that trying to update: ", find_closest_key(state), " what in q_table:", q_table[find_closest_key(state)])
            print("learning rate: ", learning_rate)
            print("q value: ", q_value)
            print("reward: ", reward)
            print("gamma: ", gamma)
            print("best q: ", best_q)

            q_table[find_closest_key(state)][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            # Set up for the next iteration
            state = next_state

            # Draw games
            env.render()

            # When episode is done, print reward
            if done or t >= MAX_TRY - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                print("-- \n\n")
                break

        # exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay


def generate_q_states(start_index, end_index, window_size):
    q_states = []    
    q_states.append(start_index) # append start index

    a = start_index[0]
    b = start_index[1]
    c = start_index[2]

    while a <= end_index[0]:
        b = start_index[1]
        while b + window_size < end_index[1]:
            c = start_index[2]
            while c + window_size < end_index[2]:
                c = c + window_size   
                # q_states = np.append(q_states, [a, b, c])
                q_states.append([a, b, c])    
            b = b + window_size
        a = a + 1
    
    return q_states


if __name__ == "__main__":
    env = gym.make("NayaaStroke-v0")
    MAX_EPISODES = 10 #9999
    MAX_TRY = 1000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6

    # num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    # print("num_box:", num_box)
    print("env.action_space.n:", env.action_space.n)
    print("env.observation_space.shape:", env.observation_space.shape)
    print("observation sample: " + str(env.observation_space.sample()))
    # q_table = np.zeros(num_box + (env.action_space.n,))

    # Create the Q-table
    q_table = {}

    q_states = generate_q_states([0, 0, 0], [1, 800, 1500], 5)

    # Initialize the Q-values for each state-action pair to 0
    for state in q_states:
        # convert list to tuple
        state = tuple(state)
        q_table[state] = {}
        for action in range(env.action_space.n):
            q_table[state][action] = 0.01 #random.uniform(0, 1)

    # print(q_table)
    simulate()
