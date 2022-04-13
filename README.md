
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
	- Returns a batch of wav files, referencing datapoints with indexes between `start` and `end` 
	 - **Parameters**: 
		 - `start` - index of starting datapoint for batch
		 - `end` - index of ending datapoint for batch
	- **Returns**: `wavs`, `labels`
		- `wavs` - $N * L$ dimensional tensor, where $N$ = number of indexes (ie. batch size) and $L$ = length of wav file in frames
		- `labels` - $P$ dimension tensor with one-hot vector of phoneme label, where $P$ is total number of phonemes in corpus
		
- `DataLoader.get_batch_phones(start, end)`
	- Returns a batch of wav files, referencing datapoints with indexes between `start` and `end`, including	phoneme label
	- **Parameters**: 
		 - `start` - index of starting datapoint for batch
		 - `end` - index of ending datapoint for batch
	- **Returns**: `wavs`, `labels`, `phone`
		- `wavs` - $N * L$ dimensional tensor, where $N$ = number of indexes (ie. batch size) and $L$ = length of wav file in frames
		- `labels` - $P$ dimension tensor with one-hot vector of phoneme label, where $P$ is total number of phonemes in corpus
		- 	`phone` - string representation of phoneme

- `DataLoader.get_batch_time(start, end)`
	- Returns ***only*** time points, referencing datapoints with indexes between `start` and `end`, 
	- **Parameters**: 
		 - `start` - index of starting datapoint for batch
		 - `end` - index of ending datapoint for batch
	- **Returns**: `time`
		- `time` - list indicating timepoints 

- `DataLoader.get_batch_wavname(start, end)`
	- Returns ***only*** wav names, referencing datapoints with indexes between `start` and `end`, 
	- **Parameters**: 
		 - `start` - index of starting datapoint for batch
		 - `end` - index of ending datapoint for batch
	- **Returns**: `wav_name``
		- `wav_name` - list indicating wav file names 

- `DataLoader.transform(wav)`
	- Transform wav file according to transformation defined in `DataLoader` instance (ie. MFCC)
	 - **Parameters**: 
		 - `wav` - $N * 1 * L$ tensor of wav file, where $N$ = batch size and $L$ = length of wav file in frames
	- **Returns**: `transformed_wav`
		- `transformed_wav` - $N* w * l$, where $N$ = batch size, and $w$ and $l$ indicate size of output representation




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



