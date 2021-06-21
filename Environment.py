import numpy as np
import random
from DataLoader import *
import torch

class Environment(object):

    def __init__(self, reward_file, state_file, episode_file, device=None):
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

        self.device = device

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
                self.rewards[line.split('\t')[0]] = torch.tensor(np.array(line.split('\t')[1:], dtype=np.double),device = self.device)

    def load_states(self, STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data[0].split('\t')
            assert header[0] == 'state'
            self.n_dims = len(header)-1

            for line in input_data[1:]:
                assert line.split('\t')[0] not in self.states.keys()
                self.states[line.split('\t')[0]] = torch.tensor(np.array(line.split('\t')[1:], dtype=np.double),device = self.device)

    def load_episodes(self, EPISODE_FILE):
        with open(EPISODE_FILE, 'r') as f:
            input_data = f.read().splitlines()
            self.n_episodes = len(input_data)

            self.episodes = [l.split(' ')[1:] for l in input_data]
            self.episode_lengths = [len(e) for e in self.episodes]

    def get_state_str(self):
        return self.current_state

    def simulation_finished(self):
        return False

class AcousticsGame1D(Environment):

    def __init__(self, reward_file, state_file, episode_file, location_file, transition_file, non_move_gap, wait_time, mode, device=None):
        super().__init__(reward_file, state_file, episode_file, device)
        self.current_location = ''
        self.n_locations=0
        self.locations = {}

        self.transitions = {}
        self.n_moves = non_move_gap
        self.current_timepoint = 0
        self.n_waittime =wait_time

        self.load_locations(location_file)
        self.load_transitions(transition_file)
        self.new_state = True

        self.action_timepoints = list(range(0, self.n_timepoints, self.n_moves))

        if mode=='correct':
            self.step=self.correct_step

        elif mode=='oneshot':
            self.step=self.oneshot_step

        elif mode=='multiple':
            self.step = self.multiple_step

        else:
            raise(AssertionError, 'mode not implemented')

    def is_new_state(self):
        return self.new_state


    def load_states(self, STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data.pop(0).split('\t')



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
                    self.states[state_name] = torch.tensor(current_array,device = self.device)

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
                self.locations[line.split('\t')[0]] = torch.tensor(np.array(line.split('\t')[1:], dtype=np.double),device = self.device)

    def load_transitions(self, TRANSITION_FILE):
        with open(TRANSITION_FILE, 'r') as f:
            input_data = f.read().splitlines()
            header = input_data[0].split('\t')
            assert header[0] == 'location'

            for line in input_data[1:]:
                assert line.split('\t')[0] not in self.states.keys()
                self.transitions[line.split('\t')[0]] = line.split('\t')[1:]


    def oneshot_step(self, action):
        if self.current_timepoint >= self.n_waittime:
            reward = self.rewards[self.current_state+'_'+self.current_location][action]
            if action>=2:
                self.current_timepoint = self.n_timepoints-1
        else:
            reward = 0
        self.advance_state(action)
        return reward

    def correct_step(self, action):
        if self.current_timepoint >= self.n_waittime:
            reward = self.rewards[self.current_state+'_'+self.current_location][action]
            if reward==1:
                self.current_timepoint = self.n_timepoints-1
        else:
            reward = 0
        self.advance_state(action)
        return reward

    def multiple_step(self, action):
        if self.current_timepoint >= self.n_waittime:
            reward = self.rewards[self.current_state+'_'+self.current_location][action]
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
            self.new_state = True

        else:
            self.current_timepoint +=1
            self.new_state = False

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
        self.current_location = random.choice(list(self.locations.keys()))

    def get_location_str(self):
        return self.current_location

    def is_action_timepoint(self):
        return  self.current_timepoint in self.action_timepoints

    def is_eps_update(self):
        return False


class AcousticsGame2DConv(AcousticsGame1D):

    def __init__(self, reward_file, state_file, episode_file, location_file, transition_file, non_move_gap, wait_time,
                 mode,  total_reps, device=None):

        self.rep = 0
        self.total_reps = total_reps-1


        super().__init__(reward_file, state_file, episode_file, location_file, transition_file, non_move_gap, wait_time, mode, device)

    def initiate_environment(self):
        self.current_location = random.choice(list(self.locations.keys()))
        self.current_episode = self.episodes[self.current_episode_num]
        self.current_state = self.current_episode[self.current_state_num]

        self.new_state = True

    def advance_episode(self):
        self.current_episode_num += 1
        self.current_episode = self.episodes[self.current_episode_num]
        self.current_state_num = 0
        self.current_state = self.current_episode[self.current_state_num]
        self.current_location = random.choice(list(self.locations.keys()))

    def get_state(self):

        if self.current_state is not None:
            return self.states[self.current_state].float(), self.locations[self.current_location].float()
        else:
            return None, None

    def get_aud_dims(self):
        h = self.states[self.episodes[0][0]].size()[0]
        w = self.states[self.episodes[0][0]].size()[1]

        return w, h


    def advance_state(self, action):
        if action < 99:
            self.current_location = self.transitions[self.current_location][action]

        self.rep += 1
        if self.rep >= self.total_reps:
            self.rep = 0
            self.current_state_num+=1
            if self.current_state_num == len(self.current_episode):
                self.current_state = None
            else:
                self.current_state = self.current_episode[self.current_state_num]

    def oneshot_step(self, action):
        if self.rep >= self.n_waittime:
            reward = self.rewards[self.current_state+'_'+self.current_location][action]
            if action>=2:
                self.rep = self.total_reps-1
        else:
            reward = 0
        self.advance_state(action)
        return reward

    def correct_step(self, action):
        if self.rep >= self.n_waittime:
            reward = self.rewards[self.current_state+'_'+self.current_location][action]
            if reward==1:
                self.rep = self.total_reps-1
        else:
            reward = 0
        self.advance_state(action)
        return reward

    def multiple_step(self, action):
        if self.rep >= self.n_waittime:
            reward = self.rewards[self.current_state+'_'+self.current_location][action]
        else:
            reward = 0
        self.advance_state(action)
        return reward

class AcousticsGame2DConvCHT(AcousticsGame2DConv):



    def __init__(self, reward_file, state_file, episode_file, location_file, transition_file, wav_file, non_move_gap, wait_time,
                 mode, total_reps, device=None, acoustic_params =('mfcc', 16000, 0.2, 400, 400, 160, 13, True),max_episodes = 50000 ):
        self.rep = 0
        self.total_reps = total_reps-1
        self.current_section_num = 0

        self.acoustic_params=acoustic_params
        self.wav_file = wav_file
        super().__init__(reward_file, state_file, episode_file, location_file, transition_file, non_move_gap, wait_time, mode, total_reps, device)

        self.step=self.cht_step
        self.n_episodes = max_episodes



    def load_states(self, STATE_FILE):

        transform_type, sr, phone_window_size, n_fft, spec_window_length, spec_window_hop, n_mfcc, log_mels = self.acoustic_params

        data = GameDataLoader(STATE_FILE, self.wav_file, self.device, transform_type=transform_type, sr = sr, phone_window_size = phone_window_size,
                 n_fft = n_fft, spec_window_length = spec_window_length, spec_window_hop = spec_window_hop, n_mfcc = n_mfcc, log_mels=log_mels)

        self.states = data.get_transformed_states()
        self.n_timepoints, _ = self.states[list(self.states.keys())[0]].size()



    def load_episodes(self, EPISODE_FILE):
        with open(EPISODE_FILE, 'r') as f:
            input_data = f.read().splitlines()
            self.n_episodes = len(input_data)
            self.sections=[]
            self.episodes=[]

            for l in input_data:
                if l[0:3] == '###':
                    _, _, section_max, _, section_need, _, section_of, _, section_threshold, _, eps_update = l.split(' ')
                    section_max = int(section_max)
                    section_need = int(section_need)
                    section_of = int(section_of)
                    section_threshold = int(section_threshold)
                    self.sections.append([section_max, section_need, section_of, section_threshold, eps_update])
                    self.episodes.append([])
                    ### max 100 need 3 of 3 threshold 10
                #this is the start of a section

                else:
                    self.episodes[-1].append(l.split(' ')[1:])
            self.episode_lengths = [[len(f) for f in e] for e in self.episodes]
            self.section_lengths = [len(e) for e in self.episodes]
            self.n_sections = len(self.section_lengths)

    def initiate_environment(self):
        self.current_section_length = self.section_lengths[self.current_section_num]
        self.current_episode_num = random.randint(0, self.current_section_length-1)
        self.current_section = self.episodes[self.current_section_num]
        self.eps_update_num = 0

        self.current_location = random.choice(list(self.locations.keys()))
        self.current_episode = self.current_section[self.current_episode_num]
        self.current_state = self.current_episode[self.current_state_num]

        self.new_state = True

        self.reward_memory=[]
        self.d_memory = [0,0,0,0] # hit, false alarm, miss, correct negative

        self.section_max, self.section_need, self.section_of, self.section_threshold, eps_update = self.sections[self.current_section_num]
        if eps_update == 'na':
            self.should_eps_update = False
        else:
            self.should_eps_update = True
            self.eps_update_num = int(eps_update)
        self.time_in_section=0

        self.is_finished=False

        self.change = False
        self.correct_headturn = None
        self.accumulated_reward = 0

    def advance_episode(self):
        # need to check if section threholds have been reached
        if self.change:
            if self.accumulated_reward >= self.section_threshold:
                success = True
                d_ind = 0
            else:
                success = False
                d_ind = 2
        else:
            if self.accumulated_reward == 10:
                success = True
                d_ind = 3
            else:
                success = False
                d_ind = 1


        #print(str(self.change), str(self.accumulated_reward), success)


        if len(self.reward_memory) < self.section_of:
            self.reward_memory.append(success)
            self.d_memory[d_ind]+=1

            #continue

        else:
            self.reward_memory.pop(0)
            self.reward_memory.append(success)
            self.d_memory[d_ind] += 1
            successes = sum(self.reward_memory)

            if successes >= self.section_need or self.time_in_section >= self.section_max:
                self.debug()
                print('success')
                print(sum(self.reward_memory))
                print('hit:', self.d_memory[0], 'alarm:', self.d_memory[1], 'miss:', self.d_memory[2], 'correct negative:', self.d_memory[3])
                print(self.change)
                print(self.correct_headturn)
                print(success)
                if self.current_section_num != self.n_sections -1:
                    print('moving to new section')
                    self.current_section_num +=1
                    self.current_section_length = self.section_lengths[self.current_section_num]
                    self.current_section = self.episodes[self.current_section_num]
                    self.section_max, self.section_need, self.section_of, self.section_threshold, eps_update = self.sections[self.current_section_num]
                    if eps_update =='na':
                        self.should_eps_update = False
                    else:
                        self.should_eps_update = True
                        self.eps_update_num = int(eps_update)
                    self.time_in_section = 0
                    self.reward_memory = []
                    self.d_memory = [0, 0, 0, 0]
                else:
                    self.is_finished=True

        # continuing in this section
        self.current_episode_num = random.randint(0, self.current_section_length - 1)
        self.current_state_num = 0
        self.current_episode = self.current_section[self.current_episode_num]
        self.current_state = self.current_episode[self.current_state_num]
        self.time_in_section +=1
        self.change = False
        self.correct_headturn = None
        self.accumulated_reward = 0

    def is_eps_update(self):
        return False


    def get_state(self):

        if self.current_state is not None:
            return self.states[self.current_state[1:]].float(), self.locations[self.current_location].float()
        else:
            return None, None

    def get_aud_dims(self):
        h = self.states[self.episodes[0][0][0][1:]].size()[0]
        w = self.states[self.episodes[0][0][0][1:]].size()[1]

        return w, h


    def advance_state(self, action):
        if action < 99:
            self.current_location = self.transitions[self.current_location][action]

        self.rep += 1

        if self.rep >= self.total_reps:
            self.rep = 0
            self.current_state_num+=1
            if self.current_state_num == len(self.current_episode):
                self.current_state = None
            else:
                self.current_state = self.current_episode[self.current_state_num]

    def cht_step(self, action):
        if self.rep >= self.n_waittime:
            reward = self.rewards[self.current_state[1:]+'_'+self.current_location][action]
        else:
            reward = 0

        if self.current_state[3] == 'r': #TODO: Update this as variable
            self.change = True

        self.accumulated_reward +=reward
        self.advance_state(action)
        return reward

    def simulation_finished(self):
        return self.is_finished

    def debug(self):
        print('-------')
        print('section parameters: ', str(self.section_max), str(self.section_need), str(self.section_of), str(self.section_threshold))
        print('episode number:',str(self.current_episode_num))
        print('time in section:',str(self.time_in_section))
        print('current_section:',str(self.current_section))
        print( 'reward memory:',str(self.reward_memory))

    def is_eps_update(self):
        return self.should_eps_update

    def get_eps_update_num(self):
        return self.eps_update_num


class AcousticsGame2DConvFromFile(AcousticsGame2DConv):

    def __init__(self, reward_file, state_file, episode_file, location_file, transition_file, wav_file, non_move_gap, wait_time,
                 mode, total_reps, device=None, acoustic_params =('mfcc', 16000, 0.2, 400, 400, 160, 13, True), ):


        self.acoustic_params=acoustic_params
        self.wav_file = wav_file
        super().__init__(reward_file, state_file, episode_file, location_file, transition_file, non_move_gap, wait_time, mode, total_reps, device)

    def load_states(self, STATE_FILE):

        transform_type, sr, phone_window_size, n_fft, spec_window_length, spec_window_hop, n_mfcc, log_mels = self.acoustic_params

        data = GameDataLoader(STATE_FILE, self.wav_file, self.device, transform_type=transform_type, sr = sr, phone_window_size = phone_window_size,
                 n_fft = n_fft, spec_window_length = spec_window_length, spec_window_hop = spec_window_hop, n_mfcc = n_mfcc, log_mels=log_mels)

        self.states = data.get_transformed_states()
        self.n_timepoints, _ = self.states[list(self.states.keys())[0]].size()





