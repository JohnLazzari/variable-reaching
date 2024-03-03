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

HID_DIM = 256
INP_DIM = 7
OUT_DIM = 2

class GoalRNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(GoalRNN, self).__init__()
        
        self.rnn = nn.RNN(inp_dim, hid_dim, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, x, hn, seq_lens):

        rnn_out, _ = self.rnn(x, hn)
        x = self.out_layer(rnn_out)

        return x, rnn_out

def gather_PMd_data(data):
    
    # return PMd data in an easier format
    PMd_data = []
    for reach in data:
        PMd_data.append(reach[0].T)
    return PMd_data

def create_training_data(kinematics, reach_order, reach_dir, target_on):
    
    # Get x and y values as targets
    target_values = []
    for reach in kinematics:
        targets = reach[0][:, :2] # x and y location of hand
        normalized_targets = (targets-np.min(targets))/(np.max(targets)-np.min(targets)) 
        target_values.append(normalized_targets)
    target_values = list(map(torch.FloatTensor, target_values))
    target_values = pad_sequence(target_values, batch_first=True)

    # Get training data 
    training_data = []
    for idx, reach in enumerate(kinematics):
        kin_data = reach[0][:, 3:7] # velocity and acceleration of hand
        normalized_kin_data = (kin_data-np.min(kin_data))/(np.max(kin_data)-np.min(kin_data)) 
        # get reach number array
        reach_num = reach_order[idx][0][0] / 4
        reach_num = np.expand_dims(np.repeat(reach_num, kin_data.shape[0]), axis=1)
        # get reach direction array
        reach_angle = (reach_dir[idx][0][0] + 3.14) / (6.28) 
        reach_angle = np.expand_dims(np.repeat(reach_angle, kin_data.shape[0]), axis=1)
        # append all of the data
        trial_data = np.concatenate((normalized_kin_data, reach_num, reach_angle, target_on[idx][0]), axis=1)
        training_data.append(trial_data)
    seq_lens = list(map(len, training_data))
    training_data = list(map(torch.FloatTensor, training_data))
    training_data = pad_sequence(training_data, batch_first=True)
        
    return training_data, target_values, seq_lens

def main():

    kinematics = sio.loadmat("source_data/firing_rates/MM_S1_kinematics.mat")["kinematics"]
    reach_order = sio.loadmat("source_data/firing_rates/MM_S1_reach_order.mat")["reach_order"]
    reach_dir = sio.loadmat("source_data/firing_rates/MM_S1_reach_dir.mat")["reach_dir"]
    target_on = sio.loadmat("source_data/firing_rates/MM_S1_target_on.mat")["target_on"]
    PMd_activity = sio.loadmat("source_data/firing_rates/MM_S1_PMd_fr.mat")["PMd_population"]
    PMd_activity = gather_PMd_data(PMd_activity)

    x, y, seq_lens = create_training_data(kinematics, reach_order, reach_dir, target_on)

    model = GoalRNN(INP_DIM, HID_DIM, OUT_DIM)
    checkpoint = torch.load("goal_rnn.pth")
    model.load_state_dict(checkpoint)

    with torch.no_grad():
        hn = torch.zeros(size=(1, x.shape[0], HID_DIM))
        out, act = model(x, hn, seq_lens)

    rnn_acts = []
    # Get data without padding
    for idx, reach in enumerate(act):
        rnn_acts.append(reach[:seq_lens[idx]].numpy())
    
    cond1_idx = []
    for idx, reach in enumerate(reach_dir):
        if reach[0][0] > np.pi/4-0.1 and reach[0][0] < np.pi/4+0.1:
            cond1_idx.append(idx)
    cond1_tensor = x[cond1_idx]

    '''
    cond1_activities = []
    for reach in cond1_tensor:
        with torch.no_grad():
            hn = torch.zeros(size=(1, HID_DIM))
            out, act = model(reach, hn, cond1_lens)
        cond1_activities.append(act)
    '''

    # PSTH for condition 1 (45 degrees)
    for idx in cond1_idx:
        plt.plot(np.mean(PMd_activity[idx], axis=1))
    plt.show()

    for idx in cond1_idx:
        plt.plot(np.mean(rnn_acts[idx], axis=1))
    plt.show()

    PC_pmd = PCA(n_components=3)
    # Plot first PC
    for idx in cond1_idx:
        pcs = PC_pmd.fit_transform(PMd_activity[idx])
        plt.plot(pcs[:, 0])
    plt.show()
    


if __name__ == "__main__":
    main()