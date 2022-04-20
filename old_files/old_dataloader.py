import random
import torch
import torchaudio
import math
from collections import namedtuple
Instance = namedtuple('Instance',
                                ('file', 'start', 'end','label', 'phone'))

class SpeechDataLoader(object):

    def __init__(self, segments_file, phones_file, alignments_file, wav_folder, device, transform_type='mfcc', sr = 16000, phone_window_size = 0.2,
                 n_fft = 400, spec_window_length = 400, spec_window_hop = 160, n_mfcc = 13, log_mels=True):

        self.phone_list = []
        self.segments = {}
        self.data = []
        self.wavs= {}
        self.sr = sr
        self.window_size = phone_window_size
        self.sr_window_size = math.floor(self.window_size * self.sr)

        self.device=device

        self.wav_folder = wav_folder

        self.vocab = {}

        self.load_phones(phones_file)
        self.load_segments(segments_file)
        self.load_alignments(alignments_file)

        self.w = int((sr * phone_window_size /spec_window_hop ) + 1)

        self.discarded_segments = 0

        if transform_type =='spectrogram':
            self.Spectrogram = torchaudio.transforms.Spectrogram(win_length = spec_window_length, hop_length = spec_window_hop).to(device)
            self.transform = self.transform_spectrogram
            self.h = n_fft // 2 + 1
        elif transform_type=='mfcc':
            self.Deltas = torchaudio.transforms.ComputeDeltas()
            self.Mfccs = torchaudio.transforms.MFCC(n_mfcc =n_mfcc, log_mels =log_mels, melkwargs = {'win_length':spec_window_length, 'normalized':True, 'hop_length':spec_window_hop}).to(device)
            self.transform = self.transform_mfcc
            self.h = n_mfcc*3
        else:
            raise NotImplementedError('no implementation of selected transform type')

    def get_feature_dims(self):
        return self.w, self.h

    def __len__(self):
        return len(self.data)

    def load_segments(self, file):
        f = open(file, 'r')
        for l in f.readlines():
            line = l.split(' ')
            self.segments[line[0]] = line[1:]

    def load_phones(self, file):
        f = open(file, 'r')
        self.phone_list = [p.split('\n')[0] for p in f.readlines()]
        self.n_phones=len(self.phone_list)
        for i_phone in range(self.n_phones):
            phone_vector = torch.zeros(self.n_phones)
            phone_vector[i_phone] = 1
            self.vocab[self.phone_list[i_phone]] = phone_vector

    def load_alignments(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        for l in lines:
            #print(l.split(' '))
            utt, phone_start, phone_end, _, phone = l[:-1].split(' ')[0:5]
            wav, utt_start, utt_end, = self.segments[utt]
            if wav not in self.wavs.keys():
                self.wavs[wav] = None
            if phone in self.phone_list:
                start = float(utt_start) + float(phone_start)
                end = float(utt_start) + float(phone_end)
                self.data.append(Instance(wav, start, end, self.vocab[phone], phone))


    def randomize_data(self):
        random.shuffle(self.data)

    def get_num_phones(self):
        return self.n_phones

    def get_batch(self, start, end):
        batch_size = end-start
        batch = self.data[start:end]
        wavs = self.get_wavs([b.file for b in batch])

        batch_unzipped = Instance(*zip(*batch))



        starts = torch.tensor(batch_unzipped.start)
        ends= torch.tensor(batch_unzipped.end)
        window_starts = torch.floor( (ends + starts) / 2 * self.sr) - (self.sr_window_size // 2)
        window_starts = window_starts.type(torch.int64)
        labels = batch_unzipped.label
        try:
            cut_wavs = torch.stack([ wavs[ind][:,window_starts[ind]:window_starts[ind]+self.sr_window_size] for ind in range(len(window_starts))])
            cut_wavs = cut_wavs.reshape(batch_size, self.sr_window_size)
        except RuntimeError:

            cut_wavs = torch.stack([ wavs[ind][:,window_starts[ind]:window_starts[ind]+self.sr_window_size] for ind in range(len(wavs)) if wavs[ind][:,window_starts[ind]:window_starts[ind]+self.sr_window_size].size()[1] == self.sr_window_size])
            labs = [batch_unzipped.label[ind] for ind in range(len(wavs)) if wavs[ind][:,window_starts[ind]:window_starts[ind]+self.sr_window_size].size()[1] == self.sr_window_size]
            labels = labs
            self.discarded_segments += (len(labels) - len(window_starts))
            cut_wavs = cut_wavs.reshape(len(cut_wavs), self.sr_window_size)




        return cut_wavs , labels

    def get_batch_phones(self, start, end):
        batch = self.data[start:end]
        batch_unzipped = Instance(*zip(*batch))

        wavs, labels = self.get_batch(start, end)

        return wavs,  labels, batch_unzipped.phone

    def get_phone_list(self):
        return self.phone_list

    def get_wavs(self, files):
        wavs = [self.wavs[w] for w in files]
        if None in wavs:
            for f in range(len(files)):
                if wavs[f]==None:
                    loaded_file, _ = self.load_wav(files[f])
                    self.wavs[files[f]] = loaded_file
                    wavs[f] = loaded_file

        return wavs


    def load_wav(self, file):
        waveform, sample_rate = torchaudio.load(self.wav_folder + file)
        if sample_rate != self.sr:
            raise(AssertionError)
        return waveform, sample_rate

    def load_all_wavs(self):
        pass

    def transform_spectrogram(self, x):
        return self.Spectrogram(x)

    def transform_mfcc(self, x):
        mfcc = self.Mfccs(x)
        d1 = self.Deltas(mfcc)
        d2 = self.Deltas(d1)
        return torch.cat((mfcc, d1, d2), -2)



class GameDataLoader(object):


    def __init__(self, states_file, wav_folder, device, transform_type='mfcc', sr = 16000, phone_window_size = 0.2,
                 n_fft = 400, spec_window_length = 400, spec_window_hop = 160, n_mfcc = 13, log_mels=True):

        self.sr = sr
        self.window_size = phone_window_size
        self.sr_window_size = math.floor(self.window_size * self.sr)
        self.states={}
        self.load_states(states_file, wav_folder, device)
        self.w = int((sr * phone_window_size /spec_window_hop ) + 1)


        if transform_type == 'spectrogram':
            self.Spectrogram = torchaudio.transforms.Spectrogram(win_length=spec_window_length,
                                                                 hop_length=spec_window_hop).to(device)
            self.transform = self.transform_spectrogram
            self.h = n_fft // 2 + 1
        elif transform_type == 'mfcc':
            self.Deltas = torchaudio.transforms.ComputeDeltas()
            self.Mfccs = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, log_mels=log_mels,
                                                    melkwargs={'win_length': spec_window_length,
                                                               'hop_length': spec_window_hop, 'normalized':True}).to(device)
            self.transform = self.transform_mfcc
            self.h = n_mfcc * 3
        else:
            raise NotImplementedError('no implementation of selected transform type')


        #self.load_states(states_file, wav_folder, device)

    def load_states(self, states_file, wav_folder, device):

        f = open(states_file, 'r')
        states = [l.strip('\n').split('\t') for l in f.readlines()]

        for s in states:
            state_name, mid_time, wav_file = s

            waveform, sample_rate = torchaudio.load(wav_folder + wav_file)

            if sample_rate != self.sr:
                raise(AssertionError)

            window_start = math.floor( float(mid_time) * self.sr) - (self.sr_window_size // 2)

            cut_wav = waveform[:, window_start:window_start+self.sr_window_size]
            cut_wav = cut_wav.reshape(self.sr_window_size).to(device)

            self.states[state_name] = cut_wav


    def get_states(self):
        return self.states

    def get_transformed_states(self):
        transformed_states={}
        for k in self.states.keys():
            transformed_states[k] = self.transform(self.states[k])
        return transformed_states


    def transform_spectrogram(self, x):
        return self.Spectrogram(x)

    def transform_mfcc(self, x):
        mfcc = self.Mfccs(x)
        d1 = self.Deltas(mfcc)
        d2 = self.Deltas(d1)
        return torch.cat((mfcc, d1, d2), -2)

    def get_batch(self, start, end):

        return [self.transform(self.states[d]) for d in list(self.states.keys())[start:end]]

    def get_random_batch(self, start, end):
        states = list(self.states.keys())
        random.shuffle(states)
        return [self.transform(self.states[d]) for d in states[start:end]]

    def get_feature_dims(self):
        return self.w, self.h
