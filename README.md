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
    
 




 - ```DataLoader.SpeechDataLoader(segments_file, phones_file, alignments_file, wav_folder, device, transform_type='mfcc', sr = 16000, phone_window_size = 0.2, n_fft = 400, spec_window_length = 400, spec_window_hop = 160, n_mfcc = 13, log_mels=True):```
	 - **Required Parameters**: 
		 - `segments_file` - full path to segments file
		 - `phones_file` - full path to phone file
		 - `alignments_file` - full path to aligment file
		 - `wav_folder` - name of wav folder (from current directory)
		 - `device` - pytorch device name

- `DataLoader.get_feature_dims()`
	- **Returns**: `w`, `h`
		- `w` - width of transformed wav features (ie. MFCC coefficients)
		- `h` - height of transformed wav features (ie. MFCC coefficients)
	
- `DataLoader.get_num_phones()`
	- **Returns**: `num_phones`
		- `num_phones` - number of phones used in current data set

- `DataLoader.get_batch(start, end)`
	- Returns a batch of 
	 - **Parameters**: 
		 - `start` - index of starting datapoint for batch
		 - `end` - index of ending datapoint for batch

	- **Returns**: `wavs`, `labels`
		- `wavs` - 
		- `labels` - 
	
- `DataLoader.get_batch_phones(start, end)`
	- Returns a batch of 
	 - **Parameters**: 
		 - `start` - index of starting datapoint for batch
		 - `end` - index of ending datapoint for batch

	- **Returns**: `wavs`, `labels`, `pohones`
		- `wavs` - 
		- `labels` - 

- `DataLoader.transform(wav)`
	- Returns a batch of 
	 - **Parameters**: 
		 - `wav` - index of starting datapoint for batch

	- **Returns**: `transformed_wav`
		- `transformed_wav` - 




## Environment
## NN

 - ```NN.PhonemeConvNN(kernel, sstride, w, h, n_outputs, layers = [32, 32, 32, 20], window_length = 100):```
	 - **Required Parameters**: 
		 - `kernel` - full path to segments file
		 - `sstride` - full path to phone file
		 - `w` - full path to aligment file
		 - `h` - name of wav folder (from current directory)

- `NN.PhonemeConvNN.forward(x)`:
	- Runs input through neural network, can also be called simply with `NN.PhonemeConvNN(x)`
	 - **Parameters**: 
		 - `x` - input to neural network
	- **Returns**: `y`
		- `y` - output from neural network
## DQN
## Loss
## ReplayMemory



