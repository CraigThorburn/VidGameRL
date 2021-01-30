from collections import namedtuple
import random
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
LSTMTransition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'hidden','next_hidden'))
ConvTransition = namedtuple('Transition',
                        ('state', 'loc', 'action', 'next_state', 'next_location', 'reward'))

ConvLSTMTransition = namedtuple('Transition',
                        ('state', 'location', 'action', 'next_state', 'next_location', 'reward', 'hidden','next_hidden'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def advance_episode(self):
        pass

class SequentialUpdatesReplayMemory(object):

    def __init__(self, episode_capacity):
        self.episode_capacity = episode_capacity
        self.memory = []
        self.episode_position = 0
        self.current_episode = []

    def push(self, *args):
        """Saves a transition."""
        self.current_episode.append(LSTMTransition(*args))

    def advance_episode(self):
        try:
            self.memory[self.episode_position] = self.current_episode
        except IndexError:
            self.memory.append(self.current_episode)
        self.current_episode = []
        self.episode_position = (self.episode_position + 1) % self.episode_capacity


    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class ConvReplayMemory(ReplayMemory):
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = ConvTransition(*args)
        self.position = (self.position + 1) % self.capacity

class ConvSequentialUpdatesReplayMemory(SequentialUpdatesReplayMemory):
    def push(self, *args):
        self.current_episode.append(ConvLSTMTransition(*args))