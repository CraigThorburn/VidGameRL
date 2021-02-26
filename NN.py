import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

class PhonemeConvNN(nn.Module):

    def __init__(self, kernel, sstride, w, h, n_outputs, window_length = 100):
        super(PhonemeConvNN, self).__init__()
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=kernel, stride=sstride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.linear_input_size = convw * convh * 32

        #self.spectrogram = torchaudio.transforms.Spectrogram(win_length = window_length)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel, stride=sstride)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=kernel, stride=sstride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel, stride=sstride)
        self.bn3 = nn.BatchNorm2d(32)

        self.lin1 = nn.Linear(self.linear_input_size, n_outputs)


    def forward(self, x):

        #x = self.spectrogram(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.size()[0], x.size()[1]*x.size()[2]*x.size()[3])

        x = F.softplus(self.lin1(x))

        return x