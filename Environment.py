import numpy as np
import random
import math
import torch

class Environment(object):

    def __init__(self, reward_file, state_file, episode_file):
        self.states = {}
        self.rewards = {}
        self.episodes = []
        self.n_actions = 0
        self.n_states = 0
        self.n_dims = 0
        self.n_episodes = 0
        self.episode_lengths = 0

        self.current_episode_num = 0
        self.current_episode = []
        self.current_state_num = 0
        self.current_state = ''

        self.load_rewards(reward_file)
        self.load_states(state_file)
        self.load_episodes(episode_file)

    def get_n_actions(self):
        return self.n_actions

    def get_n_states(self):
        return self.n_states

    def get_n_dims(self):
        return self.n_dims

    def get_n_episodes(self):
        return self.n_episodes

    def validate_environment(self):
        assert self.states != {}
        assert self.rewards != {}
        assert self.episodes !=[]

    def get_current_episode_length(self):
        return len(self.current_episode)

    def load_rewards(self, REWARD_FILE):
        with open(REWARD_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data[0].split('\t')
            assert header[0] == 'state'
            self.n_actions = len(header) - 1
            self.n_states = len(input_data)-1

            for line in input_data[1:]:
                assert line.split('\t')[0] not in self.rewards.keys()
                self.rewards[line.split('\t')[0]] = torch.tensor(np.array(line.split('\t')[1:], dtype=np.double))

    def load_states(self, STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data[0].split('\t')
            assert header[0] == 'state'
            self.n_dims = len(header)-1

            for line in input_data[1:]:
                assert line.split('\t')[0] not in self.states.keys()
                self.states[line.split('\t')[0]] = torch.tensor(np.array(line.split('\t')[1:], dtype=np.double))

    def load_episodes(self, EPISODE_FILE):
        with open(EPISODE_FILE, 'r') as f:
            input_data = f.read().splitlines()
            self.n_episodes = len(input_data)

            self.episodes = [l.split(' ')[1:] for l in input_data]
            self.episode_lengths = [len(e) for e in self.episodes]

    def get_state_str(self):
        return self.current_state


class MovementGame(Environment):

    def __init__(self, reward_file, state_file, episode_file, location_file, transition_file, stimulus_repetitions):
        super().__init__(reward_file, state_file, episode_file)
        self.current_location = ''
        self.n_locations=0
        self.locations = {}

        self.transitions = {}
        self.n_reps = stimulus_repetitions
        self.current_rep = 1

        self.load_locations(location_file)
        self.load_transitions(transition_file)

    def load_locations(self, LOCATION_FILE):
        with open(LOCATION_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data[0].split('\t')
            assert header[0] == 'location'
            self.n_location_dims = len(header)-1

            for line in input_data[1:]:
                assert line.split('\t')[0] not in self.states.keys()
                self.locations[line.split('\t')[0]] = torch.tensor(np.array(line.split('\t')[1:], dtype=np.double))

    def load_transitions(self, TRANSITION_FILE):
        with open(TRANSITION_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data[0].split('\t')
            assert header[0] == 'location'

            for line in input_data[1:]:
                assert line.split('\t')[0] not in self.states.keys()
                self.transitions[line.split('\t')[0]] = line.split('\t')[1:]


    def step(self, action):
        reward = self.rewards[self.current_state+'_'+self.current_location][action]
        if reward==1:
            self.current_rep = self.n_reps
        self.advance_state(action)
        return reward

    def advance_state(self, action):
        self.current_location = self.transitions[self.current_location][action]
        if self.current_rep == self.n_reps:
            self.current_rep = 1
            self.current_state_num+=1
            if self.current_state_num == len(self.current_episode):
                self.current_state = None
            else:
                self.current_state = self.current_episode[self.current_state_num]

        else:
            self.current_rep +=1

    def initiate_environment(self):
        self.current_location = random.choice(list(self.locations.keys()))
        self.current_episode = self.episodes[self.current_episode_num]
        self.current_state = self.current_episode[self.current_state_num]

    def get_state(self):
        if self.current_state is not None:
            return torch.cat((self.states[self.current_state], self.locations[self.current_location]),0)
        else:
            return None

    def get_n_location_dims(self):
        return self.n_location_dims

    def advance_episode(self):
        self.current_episode_num += 1
        self.current_episode = self.episodes[self.current_episode_num]
        self.current_state_num = 0
        self.current_state = self.current_episode[self.current_state_num]
        self.current_rep = 1



    def get_location_str(self):
        return self.current_location

class OneShotMovementGame(MovementGame):

    def __init__(self, reward_file, state_file, episode_file, location_file, transition_file, stimulus_repetitions):
        super.__init__(reward_file, state_file, episode_file, location_file, transition_file, stimulus_repetitions)

    def step(self, action):
        reward = self.rewards[self.current_state + '_' + self.current_location][action]
        if action == 2:
            self.current_rep = self.n_reps
        self.advance_state(action)
        return reward

class SimpleGame(Environment):

    def __init__(self, reward_file, state_file, episode_file):
        super().__init__(reward_file, state_file, episode_file)


    def advance_episode(self):
        self.current_episode_num +=1
        self.current_episode = self.episodes[self.current_episode_num]
        self.current_state_num = 0
        self.current_state = self.current_episode[self.current_state_num]


    def step(self, action):
        reward = self.rewards[self.current_state][action]
        self.advance_state()
        return reward

    def advance_state(self):
        self.current_state_num += 1
        if self.current_state_num == len(self.current_episode):
            self.current_state = None
        else:
            self.current_state = self.current_episode[self.current_state_num]

    def initiate_environment(self):
        self.current_episode = self.episodes[self.current_episode_num]
        self.current_state = self.current_episode[self.current_state_num]

    def get_state(self):
        return self.states[self.current_state]

class AcousticsGame(Environment):

    def __init__(self, reward_file, state_file, episode_file, location_file, transition_file, non_move_gap, wait_time):
        super().__init__(reward_file, state_file, episode_file)
        self.current_location = ''
        self.n_locations=0
        self.locations = {}

        self.transitions = {}
        self.n_moves = non_move_gap
        self.current_timepoint = 0
        self.n_waittime =wait_time

        self.load_locations(location_file)
        self.load_transitions(transition_file)

        self.action_timepoints = list(range(0, self.n_timepoints, self.n_moves))

    def load_states(self, STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data.pop(0).split('\t')
            print(header)



            assert header[0].split('_')[0] == 'dims'
            self.n_dims = int(header[0].split('_')[1])
            self.n_timepoints = len(header)-1

            current_array = np.zeros((self.n_dims, self.n_timepoints))
            current_dim = 0

            for i in range(len(input_data)):

                line = input_data[i].split('\t')
                if (i+1) % self.n_dims == 0:

                    # end of current state
                    state_name = line[0].split('_')[0]
                    current_array[current_dim] = np.array(line[1:], dtype=np.double)
                    self.states[state_name] = torch.tensor(current_array)

                    current_dim = 0
                    current_array = np.zeros((self.n_dims, self.n_timepoints))
                else:
                    # add to state

                    current_array[current_dim] = np.array(line[1:], dtype=np.double)
                    current_dim += 1

            # self.n_dims = len(header)-1
            #
            # current_state = input_data[1].split('_')[0]
            # for line in input_data[1:]:
            #     state_name, dim = line.split('\t')[0:1]
            #
            #     if state_name != current_state:
            #
            #     else:
            #
            #     assert line.split('\t')[0] not in self.states.keys()
            #     self.states[line.split('\t')[0]] = torch.tensor(np.array(line.split('\t')[1:], dtype=np.float32))

    def load_locations(self, LOCATION_FILE):
        with open(LOCATION_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data[0].split('\t')
            assert header[0] == 'location'
            self.n_location_dims = len(header)-1

            for line in input_data[1:]:
                assert line.split('\t')[0] not in self.states.keys()
                self.locations[line.split('\t')[0]] = torch.tensor(np.array(line.split('\t')[1:], dtype=np.double))

    def load_transitions(self, TRANSITION_FILE):
        with open(TRANSITION_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data[0].split('\t')
            assert header[0] == 'location'

            for line in input_data[1:]:
                assert line.split('\t')[0] not in self.states.keys()
                self.transitions[line.split('\t')[0]] = line.split('\t')[1:]


    def step(self, action):
        if self.current_timepoint > self.n_waittime:
            reward = self.rewards[self.current_state+'_'+self.current_location][action]
            if action==2:
                self.current_timepoint = self.n_timepoints-1
        else:
            reward = 0
        self.advance_state(action)
        return reward

    def advance_state(self, action):
        if action < 99:
            self.current_location = self.transitions[self.current_location][action]


        if self.current_timepoint == self.n_timepoints-1:
            self.current_timepoint = 0
            self.current_state_num+=1
            if self.current_state_num == len(self.current_episode):
                self.current_state = None
            else:
                self.current_state = self.current_episode[self.current_state_num]

        else:
            self.current_timepoint +=1

    def initiate_environment(self):
        self.current_location = random.choice(list(self.locations.keys()))
        self.current_episode = self.episodes[self.current_episode_num]
        self.current_state = self.current_episode[self.current_state_num]

    def get_state(self):
        if self.current_state is not None:
            return torch.cat((self.states[self.current_state][:,self.current_timepoint], self.locations[self.current_location]),0).float()
        else:
            return None

    def get_n_location_dims(self):
        return self.n_location_dims

    def advance_episode(self):
        self.current_episode_num += 1
        self.current_episode = self.episodes[self.current_episode_num]
        self.current_state_num = 0
        self.current_state = self.current_episode[self.current_state_num]
        self.current_timepoint = 1

    def get_location_str(self):
        return self.current_location

    def is_action_timepoint(self):
        return  self.current_timepoint in self.action_timepoints