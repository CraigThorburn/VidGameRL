import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_NN(nn.Module):

    def __init__(self, inputs,  outputs, layers):

        super(DQN_NN, self).__init__()
        self.lin0 = nn.Linear(inputs,20)

        #self.midlayers = []
        #for l in layers:
        #    self.midlayers.append(nn.Linear(l[0], l[1]))
        self.mid1 = nn.Linear(20,20)
        self.mid2 = nn.Linear(20, 20)
        #self.mid2 = nn.Linear(2,2)
        self.linfinal = nn.Linear(20,outputs)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        #def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #    return (size - (kernel_size - 1) - 1) // stride  + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        #linear_input_size = convw * convh * 32
        #self.head = nn.Linear(inputs, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):


        #x = torch.from_numpy(x).float()
        x = F.softplus(self.lin0(x))

        x = F.softplus(self.mid1(x))
        x = F.softplus(self.mid2(x))
       # x = F.relu(self.mid2(x))

        #for l in self.midlayers:
        #    x = F.relu(l(x))
        x = F.softplus(self.linfinal(x))
        #CHANGE: relu -> softplus

        return x # self.head(x.view(x.size(0), -1))
    #TODO: Look closer at architecture


class DQN_LSTM(nn.Module):

    def __init__(self, inputs,  outputs, layers,device):

        super(DQN_LSTM, self).__init__()
        self.lin0 = nn.Linear(inputs,20)
        self.mid1 = nn.Linear(20,40)
        self.lstm = nn.LSTM(40, 40, 1)
        #self.mid2 = nn.Linear(40,20)
        self.linfinal = nn.Linear(40,outputs)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, x_lengths, hidden):

        x = self.lin0(x)
        x = F.softplus(x)
        x = F.softplus(self.mid1(x))


        if x.dim() == 1:
            x = x.reshape(1,1,40)
            x, hidden = self.lstm(x, hidden)

        else:
            # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)

            # now run through LSTM
            x, hidden = self.lstm(x, hidden)

            # undo the packing operation
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = F.softplus(x)
      #  x = F.softplus(self.mid2(x))
        x = F.softplus(self.linfinal(x))



        return x, hidden


class DQN_NN_conv(nn.Module):

    def __init__(self, h, w, inputs, outputs, kernel = 5, sstride = 2, layers = [16, 32, 32, 20], freeze_convolution=False):
        super(DQN_NN_conv, self).__init__()
        self.conv1channels, self.conv2channels, self.conv3channels, self.mid_size = layers
        self.conv1 = nn.Conv2d(1, self.conv1channels, kernel_size=kernel, stride=sstride)
        self.bn1 = nn.BatchNorm2d(self.conv1channels)
        self.conv2 = nn.Conv2d(self.conv1channels, self.conv2channels, kernel_size=kernel, stride=sstride)
        self.bn2 = nn.BatchNorm2d(self.conv2channels)

        if freeze_convolution:
            self.conv1.bias.requires_grad=False
            self.conv1.weight.requires_grad = False
            self.bn1.bias.requires_grad=False
            self.bn1.weight.requires_grad=False
            self.conv2.bias.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.bn2.bias.requires_grad=False
            self.bn2.weight.requires_grad=False

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = kernel, stride = sstride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w,kernel,sstride))
        convh = conv2d_size_out(conv2d_size_out(h,kernel,sstride))
        linear_input_size = convw * convh * self.conv2channels

        self.head1 = nn.Linear(linear_input_size+inputs, self.mid_size)
        self.head2 = nn.Linear(self.mid_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x_aud, x_loc):

        if len(x_aud.size()) == 2:
            # not batch
            x_aud = x_aud.reshape(1,1,x_aud.size()[0], x_aud.size()[1])
            x_aud = F.relu(self.bn1(self.conv1(x_aud)))
            x_aud = F.relu(self.bn2(self.conv2(x_aud)))
            x = torch.cat((x_aud.flatten(), x_loc))
        else:
            # batch
            x_aud = x_aud.reshape(x_aud.size()[0],1,x_aud.size()[1], x_aud.size()[2])
            x_aud = F.relu(self.bn1(self.conv1(x_aud)))
            x_aud = F.relu(self.bn2(self.conv2(x_aud)))
            x = torch.cat((x_aud.reshape(x_aud.size()[0],x_aud.size()[1]*x_aud.size()[2]*x_aud.size()[3]) , x_loc), 1)

        x = F.softplus(self.head1(x))
        x = F.softplus(self.head2(x))


        return x



class DQN_NN_conv_pretrain(nn.Module):

    def __init__(self, h, w, inputs, outputs, kernel = 5, sstride = 2, layers = [16, 32, 32, 20], freeze_convolution=False, n_phone_layer=39):
        super(DQN_NN_conv_pretrain, self).__init__()
        self.conv1channels, self.conv2channels, self.conv3channels, self.mid_size = layers


        self.conv1 = nn.Conv2d(1, self.conv1channels, kernel_size=kernel, stride=sstride)
        self.bn1 = nn.BatchNorm2d(self.conv1channels)
        self.conv2 = nn.Conv2d(self.conv1channels, self.conv2channels, kernel_size=kernel, stride=sstride)
        self.bn2 = nn.BatchNorm2d(self.conv2channels)
        self.conv3 = nn.Conv2d(self.conv2channels, self.conv3channels, kernel_size=kernel, stride=sstride)
        self.bn3 = nn.BatchNorm2d(self.conv3channels)

        if freeze_convolution:
            self.conv1.bias.requires_grad=False
            self.conv1.weight.requires_grad = False
            self.bn1.bias.requires_grad=False
            self.bn1.weight.requires_grad=False
            self.conv2.bias.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.bn2.bias.requires_grad=False
            self.bn2.weight.requires_grad=False
            self.conv3.bias.requires_grad = False
            self.conv3.weight.requires_grad = False
            self.bn3.bias.requires_grad=False
            self.bn3.weight.requires_grad=False

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = kernel, stride = sstride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.linear_input_size = convw * convh * self.conv3channels

        self.lin1 = nn.Linear(self.linear_input_size, n_phone_layer)

        self.head1 = nn.Linear(n_phone_layer + inputs,self.mid_size)
        self.head2 = nn.Linear(self.mid_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x_aud, x_loc):

        if len(x_aud.size()) == 2:
            # not batch
            x_aud = x_aud.reshape(1,1,x_aud.size()[0], x_aud.size()[1])
            x_aud = F.relu(self.bn1(self.conv1(x_aud)))
            x_aud = F.relu(self.bn2(self.conv2(x_aud)))
            x_aud = F.relu(self.bn2(self.conv3(x_aud)))

            x = x_aud.reshape(1, x_aud.size()[1] * x_aud.size()[2] * x_aud.size()[3])
            x_aud = F.relu(self.lin1(x_aud))
            x = torch.cat((x_aud, x_loc))
        else:
            # batch
            x_aud = x_aud.reshape(x_aud.size()[0],1,x_aud.size()[1], x_aud.size()[2])
            x_aud = F.relu(self.bn1(self.conv1(x_aud)))
            x_aud = F.relu(self.bn2(self.conv2(x_aud)))
            x_aud = F.relu(self.bn2(self.conv3(x_aud)))

            x_aud = x_aud.reshape(x_aud.size()[0], x_aud.size()[1] * x_aud.size()[2] * x_aud.size()[3])
            x_aud = F.relu(self.lin1(x_aud))
            x = torch.cat((x_aud , x_loc), 1)

        x = F.softplus(self.head1(x))
        x = F.softplus(self.head2(x))


        return x


class DQN_convLSTM(nn.Module):

    def __init__(self, h, w, inputs, outputs, kernel = 5, sstride = 2):

        super(DQN_convLSTM, self).__init__()


        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = kernel, stride = sstride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w,kernel,sstride))
        convh = conv2d_size_out(conv2d_size_out(h,kernel,sstride))
        self.linear_input_size = convw * convh * 16

        lstm_size = 40
        lin_size = 20


        self.conv1 = nn.Conv2d(1, 8, kernel_size=kernel, stride=sstride)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel, stride=sstride)
        self.bn2 = nn.BatchNorm2d(16)

        self.trans = nn.Linear(self.linear_input_size + inputs,lstm_size)

        self.lstm = nn.LSTM(lstm_size, lstm_size, 1, batch_first=True)

        self.lin1 = nn.Linear(lstm_size,lin_size)
        self.lin2 = nn.Linear(lin_size,outputs)



    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x_aud, x_loc, x_lengths, hidden):

        if len(x_aud.size()) == 2:
            # not batch
            x_aud = x_aud.reshape(1,1,x_aud.size()[0], x_aud.size()[1])
            x_aud = F.relu(self.bn1(self.conv1(x_aud)))
            x_aud = F.relu(self.bn2(self.conv2(x_aud)))
            x = torch.cat((x_aud.flatten(), x_loc))

            x = F.softplus(self.trans(x)).reshape(1,1,40)

            x, hidden = self.lstm(x, hidden)

        else:
            # batch
            N, L, w, h = x_aud.size()
            x_aud = x_aud.reshape(N * L, 1, w, h)
            x_aud = F.relu(self.bn1(self.conv1(x_aud)))
            x_aud = F.relu(self.bn2(self.conv2(x_aud)))

            x_aud = x_aud.reshape(N, L, self.linear_input_size)
            x = torch.cat((x_aud , x_loc), 2)
            x = F.softplus(self.trans(x))

            # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)

            # now run through LSTM
            x, hidden = self.lstm(x, hidden)

            # undo the packing operation
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = F.softplus(self.lin1(x))
        x = F.softplus(self.lin2(x))

        return x, hidden
