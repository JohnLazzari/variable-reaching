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

def inverse_cca(rnn_act, pmd_act):

    # TODO make sure shapes make sense, across other files too

    #First filter the agent's activity with 20ms gaussian as done with experimental activity during preprocessing
    A_agent = gaussian_filter1d(rnn_act, 5, axis=0)

    #Reduce the activity using PCA to the first 10 components
    PC_agent = PCA(n_components=20)
    PC_exp = PCA(n_components=20)

    A_exp = PC_exp.fit_transform(pmd_act)
    A_agent = PC_agent.fit_transform(A_agent)

    #Do the CCA
    cca = CCA(n_components=20)
    U_c, V_c = cca.fit_transform(A_exp, A_agent)

    sum = 0
    for k in range(20):
        sum = sum + np.corrcoef(U_c[:, k], V_c[:, k])[0, 1]
    average = sum / 20
    U_prime = cca.inverse_transform(V_c)

    '''
    for k in range(10):
        if k==0:
            plt.plot(A_exp[:,9-k]/np.max(A_exp[:,9-k]) + k*4, linewidth=1.5, c = 'k')
            plt.plot(U_prime[:,9-k]/np.max(A_exp[:,9-k]) + k*4, linewidth= 1.5, c=(50/255, 205/255, 50/255), label= 'Network Reconstruction')
        else:
            plt.plot(A_exp[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 4, linewidth=1.5, c='k')
            plt.plot(U_prime[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 4, linewidth=1.5, c=(50 / 255, 205 / 255, 50 / 255))

    plt.ylabel('Reconstructed PMd Population Activity', size=14)
    plt.yticks([])
    plt.title(f"Inverse CCA")
    plt.show()

    #Now plot the PCs on the same plot here
    ax = plt.figure(figsize= (6,6), dpi=100).add_subplot(projection='3d')
    ax.plot(A_exp[:,0], A_exp[:, 1], A_exp[:, 2], c = 'k')
    ax.plot(U_prime[:,0], U_prime[:, 1], U_prime[:, 2], c=(50/255, 205/255, 50/255))

    # Hide grid lines
    ax.grid(False)
    plt.grid(b=None)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    plt.title(f"PC plot")
    plt.show()
    '''

    sum = 0
    for k in range(20):
        sum = sum + np.corrcoef(A_exp[:, k], U_prime[:, k])[0, 1]
    average = sum / 20

    print(f"Correlation for Train Condition {1}", average)
    return average

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
    
    correlations = []
    for reach in range(len(rnn_acts)):
        correlation = inverse_cca(rnn_acts[reach], PMd_activity[reach])
        correlations.append(correlation)
    print(f"Average correlation coefficient of inverse CCA: {np.mean(np.array(correlations))}")


    '''
    cond1_idx = []
    for idx, reach in enumerate(reach_dir):
        if reach[0][0] > -0.1 and reach[0][0] < 0.1:
            cond1_idx.append(idx)
    cond1_tensor = x[cond1_idx]

    cond1_lens = []
    for idx in cond1_idx:
        cond1_lens.append(seq_lens[idx])

    cond1_activities = []
    for reach in cond1_tensor:
        with torch.no_grad():
            hn = torch.zeros(size=(1, HID_DIM))
            out, act = model(reach, hn, cond1_lens)
        cond1_activities.append(act)

    plt.plot(np.mean(cond1_activities[2][:cond1_lens[2]].numpy(), axis=1))
    plt.plot(np.mean(PMd_activity[cond1_lens[2]], axis=0))
    plt.show()

    '''

if __name__ == "__main__":
    main()