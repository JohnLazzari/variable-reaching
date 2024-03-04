import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

EPOCHS = 10_000
LR = 1e-3
HID_DIM = 128
INP_DIM = 7
OUT_DIM = 2
WEIGHT_DECAY = 1e-3

class GoalRNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(GoalRNN, self).__init__()
        
        self.rnn = nn.GRU(inp_dim, hid_dim, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, x, hn, seq_lens):

        inp = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(inp, hn)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        out = F.relu(self.out_layer(rnn_out))

        return out

def create_training_data(kinematics, reach_order, reach_dir, target_on):
    
    # Get x and y values as targets
    target_values_1 = []
    target_values_2 = []
    target_values_3 = []
    for idx, reach in enumerate(kinematics):
        targets = reach[0][:, :2] # x and y location of hand
        normalized_targets = (targets-np.min(targets))/(np.max(targets)-np.min(targets)) 
        # only append reaches within the chosen condition (too many conditions and difficult to train on all)
        if reach_dir[idx][0][0] > np.pi/2-0.1 and reach_dir[idx][0][0] < np.pi/2+0.1:
            target_values_1.append(normalized_targets)
        if reach_dir[idx][0][0] > -0.1 and reach_dir[idx][0][0] < 0.1:
            target_values_2.append(normalized_targets)
        if reach_dir[idx][0][0] > -np.pi/2-0.1 and reach_dir[idx][0][0] < -np.pi/2+0.1:
            target_values_3.append(normalized_targets)
    target_values_1 = list(map(torch.FloatTensor, target_values_1))
    target_values_2 = list(map(torch.FloatTensor, target_values_2))
    target_values_3 = list(map(torch.FloatTensor, target_values_3))
    target_values_1 = pad_sequence(target_values_1, batch_first=True).cuda()
    target_values_2 = pad_sequence(target_values_2, batch_first=True).cuda()
    target_values_3 = pad_sequence(target_values_3, batch_first=True).cuda()
    all_target_values = [target_values_1, target_values_2, target_values_3]

    # Get training data 
    training_data_1 = []
    training_data_2 = []
    training_data_3 = []
    seq_lens = {}

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
        if reach_dir[idx][0][0] > np.pi/2-0.1 and reach_dir[idx][0][0] < np.pi/2+0.1:
            training_data_1.append(trial_data)
        if reach_dir[idx][0][0] > -0.1 and reach_dir[idx][0][0] < 0.1:
            training_data_2.append(trial_data)
        if reach_dir[idx][0][0] > -np.pi/2-0.1 and reach_dir[idx][0][0] < -np.pi/2+0.1:
            training_data_3.append(trial_data)

    seq_lens[0] = list(map(len, training_data_1))
    seq_lens[1] = list(map(len, training_data_2))
    seq_lens[2] = list(map(len, training_data_3))
    training_data_1 = list(map(torch.FloatTensor, training_data_1))
    training_data_2 = list(map(torch.FloatTensor, training_data_2))
    training_data_3 = list(map(torch.FloatTensor, training_data_3))
    training_data_1 = pad_sequence(training_data_1, batch_first=True).cuda()
    training_data_2 = pad_sequence(training_data_2, batch_first=True).cuda()
    training_data_3 = pad_sequence(training_data_3, batch_first=True).cuda()
    all_training_data = [training_data_1, training_data_2, training_data_3]
    
    return all_training_data, all_target_values, seq_lens
        

def main():

    kinematics = sio.loadmat("source_data/firing_rates/MM_S1_kinematics.mat")["kinematics"]
    reach_order = sio.loadmat("source_data/firing_rates/MM_S1_reach_order.mat")["reach_order"]
    reach_dir = sio.loadmat("source_data/firing_rates/MM_S1_reach_dir.mat")["reach_dir"]
    target_on = sio.loadmat("source_data/firing_rates/MM_S1_target_on.mat")["target_on"]

    x, y, seq_lens = create_training_data(kinematics, reach_order, reach_dir, target_on)

    model = GoalRNN(INP_DIM, HID_DIM, OUT_DIM).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):

        for i, (x_train, y_train) in enumerate(zip(x, y)):

            hn = torch.zeros(size=(1, x_train.shape[0], HID_DIM)).cuda()
            loss_mask = pad_sequence([torch.ones(size=(len, y_train.shape[-1])) for len in seq_lens[i]], batch_first=True).cuda()
            out = model(x_train, hn, seq_lens[i])
            out = out * loss_mask

            loss = criterion(out, y_train)
            print(f'Loss at epoch {epoch}: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "goal_rnn.pth")
    
if __name__ == "__main__":
    main()