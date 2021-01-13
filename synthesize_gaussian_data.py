gaus_peak = 10
max_freq = 3
freq_sep = 0.05
sigma = 0.15
rounding = 3
state_file = '/mnt/c/files/research/projects/vid_game/data/experimental_continuous_movement_parameterexploration/states.txt'
new_states_file = '/mnt/c/files/research/projects/vid_game/data/experimental_convolution_noncontinuous/states.txt'
states = {}

import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

with open(state_file, 'r') as f:
    input_data = f.read().splitlines()
    header = input_data.pop(0).split('\t')

    assert header[0].split('_')[0] == 'dims'
    n_dims = int(header[0].split('_')[1])
    n_timepoints = len(header) - 1

    current_array = np.zeros((n_dims, n_timepoints))
    current_dim = 0

    for i in range(len(input_data)):

        line = input_data[i].split('\t')
        if (i + 1) % n_dims == 0:

            # end of current state
            state_name = line[0].split('_')[0]
            current_array[current_dim] = np.array(line[1:], dtype=np.double)
            states[state_name] = np.array(current_array)

            current_dim = 0
            current_array = np.zeros((n_dims, n_timepoints))
        else:
            # add to state

            current_array[current_dim] = np.array(line[1:], dtype=np.double)
            current_dim += 1

num_freqs = int(max_freq/freq_sep) +1
freqs = np.linspace(0, max_freq, num_freqs)

new_states = {}

for s in states.keys():
    vector = states[s]
    states_length = 31#len(vector[0])
    state_2d = np.zeros((states_length, num_freqs))
    for dim in vector:
        state_from_this_f = np.zeros((states_length, num_freqs))
        for m_ind in range(states_length):
            out_dim = gaussian(freqs,dim[m_ind],sigma)
            state_from_this_f[m_ind] = out_dim
        state_2d = state_2d + state_from_this_f*gaus_peak
    state_2d = state_2d.round(rounding)
    new_states[s] = state_2d


outfile = open(new_states_file, 'w')
header = ['dims_'+str(num_freqs)] + header[1:]
outfile.write(''.join([h + '\t' for h in header[:-1]]) + header[-1])
outfile.write('\n')



for s in new_states.keys():
    for ind in range(num_freqs):
        d = new_states[s][:,ind]
        outfile.write(s + '_' + str(ind) + '\t'+''.join([str(i)+'\t' for i in d[:-1]])+ str(d[-1]))
        outfile.write('\n')
outfile.close()
print('done')