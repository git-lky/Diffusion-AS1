#!/usr/bin/env python
# !coding=utf-8

import random
from collections import deque
import numpy as np
import math


class Memory_swm(object):
    def __init__(self, memsize):
        self.memsize = memsize
        self.memory = deque(maxlen=self.memsize)

    def add_episode(self, epsiode):
        self.memory.append(epsiode)

    def get_batch(self, bsize, time_step):
        sampled_epsiodes = random.sample(self.memory, bsize)
        batch = []
        for episode in sampled_epsiodes:
            point = np.random.randint(0, len(episode) + 1 - time_step)
            batch.append(episode[point:point + time_step])
        return batch

    def del_episode(self):
        if len(self.memory) >= 5000:
            self.memory.pop(0)


class Memory_wind_field(object):
    def __init__(self, memsize):
        self.memsize = memsize
        self.memory_wind = deque(maxlen=self.memsize)
        # self.memory_wind_y = deque(maxlen=self.memsize)

    def add_list(self, wind_list_x, wind_list_y, wind_vector_x, wind_vector_y):
        self.memory_wind.append((wind_list_x, wind_list_y, wind_vector_x, wind_vector_y))
        # self.memory_wind_y.append(wind_list_y)

    def get_list(self, batch_size=1):
        sampled_x_lists, sampled_y_lists, sampled_x_vector, sampled_y_vector = \
        random.sample(self.memory_wind, batch_size)[0]
        # sampled_y_lists = random.sample(self.memory_wind_y, batch_size)[0]
        return sampled_x_lists, sampled_y_lists, sampled_x_vector, sampled_y_vector

    def add_list_tensor(self, wind_list_x, wind_list_y):
        self.memory_wind.append((wind_list_x, wind_list_y))
        # self.memory_wind_y.append(wind_list_y)

    def get_list_tensor(self, batch_size=1):
        sampled_x_lists, sampled_y_lists = \
        random.sample(self.memory_wind, batch_size)[0]
        # sampled_y_lists = random.sample(self.memory_wind_y, batch_size)[0]
        return sampled_x_lists, sampled_y_lists

    def clear(self):
        self.memory_wind.clear()


class Memory_cloud_field(object):
    def __init__(self, memsize):
        self.memsize = memsize
        self.memory_cloud = deque(maxlen=self.memsize)

    def add_list(self, cloud_list):
        self.memory_cloud.append(cloud_list)

    def get_list(self, batch_size=1):
        sampled_cloud_list = random.sample(self.memory_cloud, batch_size)[0]
        return sampled_cloud_list


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.memory_counter = 0
        self.buffer = deque(maxlen=buffer_size)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.memory_counter += 1

    def __len__(self):
        return len(self.buffer)
