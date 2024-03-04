# variable-reaching

source_data contains the raw, preprocessed, and firing rate/kinematic data used in the python files. To start from scratch, you can simply run save_fr.m which would filter the spikes to obtain their firing rates while separating other variables into individual folders for ease of use in python, however these files are already saved in source_data/firing_rates. 

If you are starting from the github and do not have the data, then simply download the dataset from https://crcns.org/data-sets/motor-cortex/pmd-1/about-pmd-1 and make sure the processing and source_data folders are in the same directory as the scripts. From there, you will need to first run save_fr.m, then the python scripts below can be used.

To train the RNN, you can run `python train_rnn.py` in the current folder. This will save a model called `goal_rnn.pth`. From here, running `python psth_pca.py` will run the PSTH experiment from the assignment (along with plotting the first PCs). Running `python cca.py` will run the inverse CCA experiment. Files will be saved to a results folder for each of these scripts, and the CCA file will print the average correlation across conditions at the bottom.