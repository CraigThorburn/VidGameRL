import random
import torch
import torchaudio
import math
from collections import namedtuple
Instance = namedtuple('Instance',
                                ('file', 'start', 'end','label'))

class SpeechDataLoader(object):

    def __init__(self, segments_file, phones_file, alignments_file, wav_folder, sr = 16000, window_size = 0.2):

        self.phone_list = []
        self.segments = {}
        self.data = []
        self.wavs= {}
        self.sr = sr
        self.window_size = window_size
        self.sr_window_size = math.floor(self.window_size * self.sr)

        self.wav_folder = wav_folder

        self.vocab = {}

        self.load_phones(phones_file)
        self.load_segments(segments_file)
        self.load_alignments(alignments_file)

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
                self.data.append(Instance(wav, start, end, self.vocab[phone]))


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
        cut_wavs = torch.stack([ wavs[ind][:,window_starts[ind]:window_starts[ind]+self.sr_window_size] for ind in range(len(window_starts))])

        return cut_wavs.reshape(batch_size, self.sr_window_size), batch_unzipped.label

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

