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
        self.lin0 = nn.Linear(inputs,40)
        self.mid1 = nn.Linear(40,40)
        self.lstm = nn.LSTM(40, 40, 1)
        self.mid2 = nn.Linear(40,20)
        self.linfinal = nn.Linear(20,outputs)


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


        x = F.softplus(self.mid2(x))
        x = F.softplus(self.linfinal(x))


        return x, hidden
