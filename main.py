import sys
import numpy as np
import math
import random

import gymnasium as gym
import gym_game

import matplotlib.pyplot as plt

q_states = []

in_out_vals = []
y_distance_from_start_vals = []
x_distance_to_goal_vals = []

screen_width = 1500
screen_height = 800

pen_start_point = [120, 590]
pen_start_point_env_4 = [100, 700]

goal_point = [1260, 600]
goal_point_env_4 = [1350, 180]

episode_timesteps = []
episode_rewards = []

def find_closest_key(key): 
    index_0 = key[0]  
    index_1 = key[1]
    index_2 = key[2]

    # print(" >>>>> in find closest key <<<<<")
    # print("in_out_vals: ", in_out_vals)
    # print("y_distance_from_start_vals: ", y_distance_from_start_vals)
    # print("x_distance_to_goal_vals: ", x_distance_to_goal_vals)
    # print("q_states: ", q_states)

    # get the closest number available in q states for observation 1.
    if index_0 in in_out_vals:
        index_0 = index_0
    else:
        index_0 = min(in_out_vals, key=lambda x: abs(x - index_0))

    # get the closest number available in q states for observation 2 (y distance travelled from start).
    if index_1 in y_distance_from_start_vals:
        index_1 = index_1
    else:
        index_1 = min(y_distance_from_start_vals, key=lambda x: abs(x - index_1))

    # get the closest number available in q states for observation 3 (x distance to goal).
    if index_2 in x_distance_to_goal_vals:
        index_2 = index_2
    else:
        index_2 = min(x_distance_to_goal_vals, key=lambda x: abs(x - index_2))


    return tuple([index_0, index_1, index_2])


def simulate():
    global epsilon, epsilon_decay

    for episode in range(MAX_EPISODES):

        # Init environment
        state, info = env.reset()
        state = tuple(state)
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):

            # In the beginning, do random action to learn
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[find_closest_key(state)])

            # action = np.argmax(q_table[find_closest_key(state)])
            
            # action = 0

            print("\nrun ", t,  " action: ", action)

            # Do action and get result
            next_state, reward, done, truncated, info = env.step(action)
            next_state = tuple(next_state)
            total_reward += reward

            print("run ", t, " prev state: ", state)
            print("run ", t, " state after doing action: ", next_state)

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

            print("q table state that updated: ", find_closest_key(state), " what in q_table:", q_table[find_closest_key(state)])

            # Set up for the next iteration
            state = next_state

            # Draw games
            print("run ", t, " total redering $$$$$$")
            env.render()
            print("run ", t, " total redered $$$$$$")


            # When episode is done, print reward
            if done or t >= MAX_TRY - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                print("-- \n\n")
                episode_rewards.append(total_reward)
                episode_timesteps.append(t)
                break

        # exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay


def generate_q_states(start_index, end_index, window_size):
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
    
    in_out_vals = [subarray[0] for subarray in q_states]
    y_distance_from_start_vals = [subarray[1] for subarray in q_states]
    x_distance_to_goal_vals = [subarray[2] for subarray in q_states]

    in_out_vals = sorted(set(in_out_vals))
    y_distance_from_start_vals = sorted(set(y_distance_from_start_vals))
    x_distance_to_goal_vals = sorted(set(x_distance_to_goal_vals))

    # print("in_out_vals: ", in_out_vals)
    # print("y_distance_from_start_vals: ", y_distance_from_start_vals)
    # print("x_distance_to_goal_vals: ", x_distance_to_goal_vals)

    return q_states, in_out_vals, y_distance_from_start_vals, x_distance_to_goal_vals


if __name__ == "__main__":
    env = gym.make("NayaaStroke-v0")
    MAX_EPISODES = 1000 #9999
    MAX_TRY = 100 #9999
    epsilon = 1
    epsilon_decay = 0.999 #0.999
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

    # q_states, in_out_vals, y_distance_from_start_vals, x_distance_to_goal_vals = generate_q_states([0, -210, -240], [1, 590, 1260], 30) 
    q_states, in_out_vals, y_distance_from_start_vals, x_distance_to_goal_vals = generate_q_states([0, -100, -150], [1, 700, 1350], 60) # for env 4
    print("q_states: ", q_states[0:3])


    # Initialize the Q-values for each state-action pair to 0
    for state in q_states:
        # convert list to tuple
        state = tuple(state)
        q_table[state] = {}
        for action in range(env.action_space.n):
            q_table[state][action] = random.uniform(0, 1)

    # print(q_table)
    simulate()
    
    print(list(q_table.items())[:4])
    print("q table size: ", len(q_table))
    
    # plot the reward and episode length
    plt.figure(figsize=(12,8))
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.subplot(2, 1, 2)
    plt.plot(episode_timesteps)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.show()


