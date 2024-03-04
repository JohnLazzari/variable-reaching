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
from matplotlib import colors

HID_DIM = 128
INP_DIM = 7
OUT_DIM = 2

cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),
 
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
 
         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }

cmap = colors.LinearSegmentedColormap('custom', cdict1)

class GoalRNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(GoalRNN, self).__init__()
        
        self.rnn = nn.GRU(inp_dim, hid_dim, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, x, hn, seq_lens):

        rnn_out, _ = self.rnn(x, hn)
        x = F.relu(self.out_layer(rnn_out))

        return x, -rnn_out # face activity upright for comparison

def gather_M1_data(data):
    
    # return M1 data in an easier format
    M1_data = []
    for reach in data:
        M1_data.append(reach[0].T)
    return M1_data

def plot_psth(M1_act, rnn_act, cond_list, cond_num, lens):

    for i, idx in enumerate(cond_list):
        plt.plot(np.mean(M1_act[idx], axis=1), c=cmap(i/len(cond_list)))
    plt.savefig(f'results/exp_act_psth_cond{cond_num}.png')
    plt.close()

    for i, (reach, length) in enumerate(zip(rnn_act, lens)):
        reach = gaussian_filter1d(reach, 10, axis=0)
        plt.plot(np.mean(reach[:length], axis=1), c=cmap(i/len(rnn_act)))
    plt.savefig(f'results/rnn_act_psth_cond{cond_num}.png')
    plt.close()

def plot_pca(M1_act, rnn_act, cond_list, cond_num, lens):

    PC_M1 = PCA(n_components=3)
    PC_rnn = PCA(n_components=3)

    for i, idx in enumerate(cond_list):
        pcs = PC_M1.fit_transform(M1_act[idx])
        plt.plot(pcs[:, 0], c=cmap(i/len(cond_list)))
    plt.savefig(f'results/exp_act_pca_cond{cond_num}.png')
    plt.close()

    for i, (reach, length) in enumerate(zip(rnn_act, lens)):
        reach = gaussian_filter1d(reach, 10, axis=0)
        pcs = PC_rnn.fit_transform(reach[:length])
        plt.plot(pcs[:, 0], c=cmap(i/len(rnn_act)))
    plt.savefig(f'results/rnn_act_pca_cond{cond_num}.png')
    plt.close()

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
    target_values_1 = pad_sequence(target_values_1, batch_first=True)
    target_values_2 = pad_sequence(target_values_2, batch_first=True)
    target_values_3 = pad_sequence(target_values_3, batch_first=True)
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
    training_data_1 = pad_sequence(training_data_1, batch_first=True)
    training_data_2 = pad_sequence(training_data_2, batch_first=True)
    training_data_3 = pad_sequence(training_data_3, batch_first=True)
    all_training_data = [training_data_1, training_data_2, training_data_3]
    
    return all_training_data, all_target_values, seq_lens

def main():

    kinematics = sio.loadmat("source_data/firing_rates/MM_S1_kinematics.mat")["kinematics"]
    reach_order = sio.loadmat("source_data/firing_rates/MM_S1_reach_order.mat")["reach_order"]
    reach_dir = sio.loadmat("source_data/firing_rates/MM_S1_reach_dir.mat")["reach_dir"]
    target_on = sio.loadmat("source_data/firing_rates/MM_S1_target_on.mat")["target_on"]
    M1_activity = sio.loadmat("source_data/firing_rates/MM_S1_M1_fr.mat")["M1_population"]
    M1_activity = gather_M1_data(M1_activity)

    x, y, seq_lens = create_training_data(kinematics, reach_order, reach_dir, target_on)

    model = GoalRNN(INP_DIM, HID_DIM, OUT_DIM)
    checkpoint = torch.load("goal_rnn.pth")
    model.load_state_dict(checkpoint)

    rnn_out = {}
    rnn_act = {}
    with torch.no_grad():
        for i, x_train in enumerate(x):
            hn = torch.zeros(size=(1, x_train.shape[0], HID_DIM))
            out, act = model(x_train, hn, seq_lens[i])
            rnn_out[i] = out
            rnn_act[i] = act

    # solution of network for all conditions
    for i, cond in enumerate(y):
        for j, reach in enumerate(cond):
            plt.plot(reach[:seq_lens[i][j]].numpy())
    plt.show()

    for i in range(3):
        for j, reach in enumerate(rnn_out[i]):
            plt.plot(reach[:seq_lens[i][j]].numpy())
    plt.show()

    cond1_idx = []
    cond2_idx = []
    cond3_idx = []
    for idx, reach in enumerate(M1_activity):
        if reach_dir[idx][0][0] > np.pi/2-0.1 and reach_dir[idx][0][0] < np.pi/2+0.1:
            cond1_idx.append(idx)
        if reach_dir[idx][0][0] > -0.1 and reach_dir[idx][0][0] < 0.1:
            cond2_idx.append(idx)
        if reach_dir[idx][0][0] > -np.pi/2-0.1 and reach_dir[idx][0][0] < -np.pi/2+0.1:
            cond3_idx.append(idx)

    # PSTH for condition 1 (90 degrees)
    plot_psth(M1_activity, rnn_act[0], cond1_idx, 1, seq_lens[0])
    # PSTH for condition 2 (0 degrees)
    plot_psth(M1_activity, rnn_act[1], cond2_idx, 2, seq_lens[1])
    # PSTH for condition 2 (-90 degrees)
    plot_psth(M1_activity, rnn_act[2], cond3_idx, 3, seq_lens[2])

    # Plot first PC cond 1
    plot_pca(M1_activity, rnn_act[0], cond1_idx, 1, seq_lens[0])
    # Plot first PC cond 2
    plot_pca(M1_activity, rnn_act[1], cond2_idx, 2, seq_lens[1])
    # Plot first PC cond 3
    plot_pca(M1_activity, rnn_act[2], cond3_idx, 3, seq_lens[2])

if __name__ == "__main__":
    main()