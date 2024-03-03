import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

EPOCHS = 1000
LR = 1e-3
HID_DIM = 256
INP_DIM = 7
OUT_DIM = 2
WEIGHT_DECAY = 1e-2

class GoalRNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(GoalRNN, self).__init__()
        
        self.rnn = nn.RNN(inp_dim, hid_dim, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, x, hn, seq_lens):

        x = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x, hn)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.out_layer(x)

        return x

def create_training_data(kinematics, reach_order, reach_dir, target_on):
    
    # Get x and y values as targets
    target_values = []
    for reach in kinematics:
        targets = reach[0][:, :2] # x and y location of hand
        normalized_targets = (targets-np.min(targets))/(np.max(targets)-np.min(targets)) 
        target_values.append(normalized_targets)
    target_values = list(map(torch.FloatTensor, target_values))
    target_values = pad_sequence(target_values, batch_first=True).cuda()

    # Get training data 
    training_data = []
    for idx, reach in enumerate(kinematics):
        kin_data = reach[0][:, 3:7] # velocity and acceleration of hand
        normalized_kin_data = (kin_data-np.min(kin_data))/(np.max(kin_data)-np.min(kin_data)) 
        # get reach number array
        reach_num = reach_order[idx][0][0] / 4
        reach_num = np.expand_dims(np.repeat(reach_num, kin_data.shape[0]), axis=1)
        # get reach direction array
        reach_angle = (reach_dir[idx][0][0] + 3.14) / 6.28
        reach_angle = np.expand_dims(np.repeat(reach_angle, kin_data.shape[0]), axis=1)
        # append all of the data
        trial_data = np.concatenate((normalized_kin_data, reach_num, reach_angle, target_on[idx][0]), axis=1)
        training_data.append(trial_data)
    seq_lens = list(map(len, training_data))
    training_data = list(map(torch.FloatTensor, training_data))
    training_data = pad_sequence(training_data, batch_first=True).cuda()
        
    return training_data, target_values, seq_lens

def main():

    kinematics = sio.loadmat("source_data/firing_rates/MM_S1_kinematics.mat")["kinematics"]
    reach_order = sio.loadmat("source_data/firing_rates/MM_S1_reach_order.mat")["reach_order"]
    reach_dir = sio.loadmat("source_data/firing_rates/MM_S1_reach_dir.mat")["reach_dir"]
    target_on = sio.loadmat("source_data/firing_rates/MM_S1_target_on.mat")["target_on"]

    x, y, seq_lens = create_training_data(kinematics, reach_order, reach_dir, target_on)
    loss_mask = pad_sequence([torch.ones(size=(len, y.shape[-1])) for len in seq_lens], batch_first=True).cuda()

    model = GoalRNN(INP_DIM, HID_DIM, OUT_DIM).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):

        hn = torch.zeros(size=(1, x.shape[0], HID_DIM)).cuda()
        out = model(x, hn, seq_lens)
        out *= loss_mask

        loss = criterion(out, y)
        print(f'Loss at epoch {epoch}: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "goal_rnn.pth")
    
    
if __name__ == "__main__":
    main()