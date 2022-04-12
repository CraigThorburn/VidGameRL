# VidGameRL

This github repository provides code to run reinforcement learning and supervised simulations of speech category learning.  There are two primary running scripts in this respository:

 - Reinforcement Learning - acousticgame
 - Supervised Learning - fullsupervision 

# Libraries
Auxilliary code is organized into several libraries:
 - **DataLoader**: Allows for loading and mannipulation of speech files
 - **Environment**: Manages a video game reinforcement paradigm
 - **NN**: Network classes for a phoneme classification network
 - **DQN**: Network for a deep Q reinforcement learning network
 - **Loss**: Includes classes for various loss functions during DQN training
 - **ReplayMemory**:  Class which defines replay memory for use during reinforcment Learning
## DataLoader
    DataLoader.SpeechDataLoader(segments_file, phones_file, alignments_file, wav_folder, device, transform_type='mfcc', sr = 16000, phone_window_size = 0.2, n_fft = 400, spec_window_length = 400, spec_window_hop = 160, n_mfcc = 13, log_mels=True):

    DataLoader.SpeechDataLoader()

## Environment
## NN
## DQN
## Loss
## ReplayMemory


