import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

class PhonemeConvNN(nn.Module):

    def __init__(self, kernel, sstride, w, h, n_outputs, layers = [32, 32, 32, 20], window_length = 100):
        super(PhonemeConvNN, self).__init__()
        self.conv1channels, self.conv2channels, self.conv3channels, self.mid_size = layers
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=kernel, stride=sstride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.linear_input_size = convw * convh * self.conv3channels

        print(convw)
        print(convh)
        print(self.linear_input_size)

        #self.spectrogram = torchaudio.transforms.Spectrogram(win_length = window_length)
        self.conv1 = nn.Conv2d(1, self.conv1channels, kernel_size=kernel, stride=sstride)
        self.bn1 = nn.BatchNorm2d(self.conv1channels)
        self.conv2 = nn.Conv2d(self.conv1channels, self.conv2channels, kernel_size=kernel, stride=sstride)
        self.bn2 = nn.BatchNorm2d(self.conv2channels)
        self.conv3 = nn.Conv2d(self.conv2channels, self.conv3channels, kernel_size=kernel, stride=sstride)
        self.bn3 = nn.BatchNorm2d(self.conv3channels)

        self.lin1 = nn.Linear(self.linear_input_size, n_outputs)


    def forward(self, x):

        #x = self.spectrogram(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.size()[0], x.size()[1]*x.size()[2]*x.size()[3])

        x = F.softplus(self.lin1(x))

        return x

    def get_out_from_layer(self, x, layer):

        if layer >= -4:
            x = F.relu(self.bn1(self.conv1(x)))

        if layer >= -3:
            x = F.relu(self.bn2(self.conv2(x)))

        if layer >= -2:
            x = F.relu(self.bn3(self.conv3(x)))
            x = x.reshape(x.size()[0], x.size()[1]*x.size()[2]*x.size()[3])

        if layer >= -1:
            x = F.softplus(self.lin1(x))

        return x



class PhonemeConvNN_extranodes(nn.Module):

    def __init__(self, kernel, sstride, w, h, n_outputs, layers = [32, 32, 32, 20], extra_nodes=2, window_length = 100):
        super(PhonemeConvNN_extranodes, self).__init__()
        self.conv1channels, self.conv2channels, self.conv3channels, self.mid_size = layers
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=kernel, stride=sstride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.linear_input_size = convw * convh * self.conv3channels

        print(convw)
        print(convh)
        print(self.linear_input_size)

        #self.spectrogram = torchaudio.transforms.Spectrogram(win_length = window_length)
        self.conv1 = nn.Conv2d(1, self.conv1channels, kernel_size=kernel, stride=sstride)
        self.bn1 = nn.BatchNorm2d(self.conv1channels)
        self.conv2 = nn.Conv2d(self.conv1channels, self.conv2channels, kernel_size=kernel, stride=sstride)
        self.bn2 = nn.BatchNorm2d(self.conv2channels)
        self.conv3 = nn.Conv2d(self.conv2channels, self.conv3channels, kernel_size=kernel, stride=sstride)
        self.bn3 = nn.BatchNorm2d(self.conv3channels)

        self.lin1 = nn.Linear(self.linear_input_size, n_outputs)
        self.lin1_extra = nn.Linear(self.linear_input_size, extra_nodes)


    def forward(self, x):

        #x = self.spectrogram(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.size()[0], x.size()[1]*x.size()[2]*x.size()[3])

        x_orig = F.softplus(self.lin1(x))
        x_extra = F.softplus(self.lin1_extra(x))


        return x_orig, x_extra

    def get_out_from_layer(self, x, layer, stacked_out=False):

        if layer >= -4:
            x = F.relu(self.bn1(self.conv1(x)))

        if layer >= -3:
            x = F.relu(self.bn2(self.conv2(x)))

        if layer >= -2:
            x = F.relu(self.bn3(self.conv3(x)))
            x = x.reshape(x.size()[0], x.size()[1]*x.size()[2]*x.size()[3])

        if layer >= -1:
            x_orig = F.softplus(self.lin1(x))
            x_extra = F.softplus(self.lin1_extra(x))

            if stacked_out:
                x = torch.cat((x_orig, x_extra), 1)
            else:
                return x_orig, x_extra


        return x