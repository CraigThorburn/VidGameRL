import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KERNEL = 3
STRIDE = 2

training_state_a = torch.ones(5,21)*10
training_state_b = torch.ones(5,21)*0
state = torch.cat((training_state_a, training_state_b))
location = torch.tensor((1,0))
action_values = torch.tensor((8,12))

training_data = [(state,location, action_values)]

w, h = state.size()
num_inputs = len(location)
n_actions = len(action_values)

hidden = (torch.zeros(1, 1, 40, device=device), torch.zeros(1, 1, 40, device=device))

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


model = DQN_convLSTM(h, w,num_inputs, n_actions, KERNEL, STRIDE).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.1)
loss_function = nn.SmoothL1Loss()

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    state, location, action = training_data[0]
    output, hidden = model(state, location, 21, hidden)
    print(output)


losses = []
params = []
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for tt in training_data:
        optimizer.zero_grad()
        state, location, action = tt
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance


        # Step 3. Run our forward pass.
        output, hidden = model(state, location, 21, hidden)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()

        loss = loss_function(output.flatten().double(), action.double())
        loss.backward()
        optimizer.step()
        #loss = loss.double()
        #losses.append(float(loss))


        losses.append(float(loss))


# See what the scores are after training
with torch.no_grad():
    state, location, action = training_data[0]
    output, hidden = model(state, location, 21, hidden)
    print(output)