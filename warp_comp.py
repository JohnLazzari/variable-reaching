import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from scipy.ndimage import gaussian_filter1d
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import torch.nn.functional as F

HID_DIM = 128
INP_DIM = 7
OUT_DIM = 2

class GoalRNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(GoalRNN, self).__init__()
        
        self.rnn = nn.GRU(inp_dim, hid_dim, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, x, hn, seq_lens):

        rnn_out, _ = self.rnn(x, hn)
        x = F.relu(self.out_layer(rnn_out))

        return x, rnn_out

def gather_M1_data(data):
    
    # return M1 data in an easier format
    M1_data = []
    for reach in data:
        M1_data.append(reach[0].T)
    return M1_data

def create_training_data(kinematics, reach_order, reach_dir, target_on):
    
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
    training_data_1 = pad_sequence(training_data_1, batch_first=True)
    training_data_2 = pad_sequence(training_data_2, batch_first=True)
    training_data_3 = pad_sequence(training_data_3, batch_first=True)
    all_training_data = [training_data_1, training_data_2, training_data_3]
    
    return all_training_data, seq_lens

def main():

    kinematics = sio.loadmat("source_data/firing_rates/MM_S1_kinematics.mat")["kinematics"]
    reach_order = sio.loadmat("source_data/firing_rates/MM_S1_reach_order.mat")["reach_order"]
    reach_dir = sio.loadmat("source_data/firing_rates/MM_S1_reach_dir.mat")["reach_dir"]
    target_on = sio.loadmat("source_data/firing_rates/MM_S1_target_on.mat")["target_on"]
    M1_activity = sio.loadmat("source_data/firing_rates/MM_S1_M1_fr.mat")["M1_population"]
    M1_activity = gather_M1_data(M1_activity)

    x, seq_lens = create_training_data(kinematics, reach_order, reach_dir, target_on)

    model = GoalRNN(INP_DIM, HID_DIM, OUT_DIM)
    checkpoint = torch.load("goal_rnn.pth")
    model.load_state_dict(checkpoint)

    rnn_act = {}
    with torch.no_grad():
        for i, x_train in enumerate(x):
            hn = torch.zeros(size=(1, x_train.shape[0], HID_DIM))
            out, act = model(x_train, hn, seq_lens[i])
            rnn_act[i] = act

    # Choose one condition to look through
    cond1_idx = []
    for idx, reach in enumerate(reach_dir):
        if reach[0][0] > -0.1 and reach[0][0] < 0.1:
            cond1_idx.append(idx)

    for (act, idx) in zip(rnn_act[1], cond1_idx):
        print(idx)
        plt.imshow(M1_activity[idx].T)
        plt.colorbar()
        plt.show()
        plt.imshow(act.numpy().T)
        plt.colorbar()
        plt.show()


        

if __name__ == "__main__":
    main()