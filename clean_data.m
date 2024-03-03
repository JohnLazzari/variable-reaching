% Takes raw data files from Miller lab, and outputs cleaned files:
% -Extracts individual reaching movements and associated data
% -Downsamples signals
%
% 

%% Load data

fpath = './source_data/raw/';
fname = 'MM_S1_raw.mat'; % Monkey M (MM), session 1; PMd and M1
% fname = 'MT_S1_raw.mat'; % Monkey T (MT), session 1; PMd only
% fname = 'MT_S2_raw.mat'; % Monkey T (MT), session 2; PMd only
% fname = 'MT_S3_raw.mat'; % Monkey T (MT), session 3; PMd only

load([fpath fname])

alldays.tt = trial_table;

% Check for existence of M1 data - only one animal (MM) has it
M1_present = exist('M1','var');

%% User inputs

dt = .01; % in seconds. .01 = 10ms, .02 = 20ms, etc

time_end_extend = 0.4; % in seconds; time after end of reach to continue grabbing data
time_before_extend = -0.3; % in seconds; time before reach start to start grabbing data; needs to be negative to go backwards

%% Initialize

blocks = 1; % there is only one block in this data set
correct_trials_only = 0; % All included trials are correct by default; this is legacy
col_start_time = 1; % column # in the table with experimental data for time of trial start; user shouldn't change this
col_end_time = 22;  % column # in the table with experimental data for time of trial end; user shouldn't change this

    bl_trials{1} = ones(size(trial_table,1),1);
    
    % Initialize M1_units and PMd_units
    PMd_units = cell(length(PMd.units),1);
    if M1_present
        M1_units = cell(length(M1.units),1);
    end

%% Extract data


% For PMd units
for i = 1:length(PMd.units)
    PMd_units{i} = PMd.units(i).ts;
end

% For M1 units
if M1_present
    for i = 1:length(M1.units)
        M1_units{i} = M1.units(i).ts;
    end
end

kin.pos(:,1) = cont.t;
kin.pos(:,2:3) = cont.pos;
kin.vel(:,1) = cont.t;
kin.vel(:,2:3) = cont.vel;
kin.acc(:,1) = cont.t;
kin.acc(:,2:3) = cont.acc;


% Extract neural data
tic

num_units_PMd = length(PMd_units);
if M1_present
    num_units_M1 = length(M1_units);
end

max_spk_time = max(kin.pos(:,1)); % Set max timestamp to collect spikes
    
edges = 0:dt:(max_spk_time + dt); % Defines edges for histogram; used for binning data later

% Initialize variables that will contain spikes
neural_data_temp_PMd = nan(num_units_PMd,length(edges)-1);
if M1_present
    neural_data_temp_M1 = nan(num_units_M1,length(edges)-1);
end

% Bin data using histcounts()
for nrn_num = 1:num_units_PMd
    neural_data_temp_PMd(nrn_num,:) = histcounts(PMd_units{nrn_num},edges);
    disp(['PMd unit: ' num2str(nrn_num) '/' num2str(num_units_PMd)])
end

if M1_present
    for nrn_num = 1:num_units_M1
        neural_data_temp_M1(nrn_num,:) = histcounts(M1_units{nrn_num},edges);
        disp(['M1 unit: ' num2str(nrn_num) '/' num2str(num_units_M1)])
    end
end

% IMPORTANT: 
% Spikes collected from 0 ms onward; Kinematics from 1000ms onward
% This gets rid of the spikes before kinematics  are collected
% I.e., with 10ms bins, 1000ms = 100 bins * 10ms/bin

num_bins_discard = round(1/dt); % number of bins to discard before kinematics are colletcted
bin_start = num_bins_discard + 1; % bin to start keeping track of data

neural_data_temp_PMd = neural_data_temp_PMd(:,bin_start:end);
if M1_present
    neural_data_temp_M1 = neural_data_temp_M1(:,bin_start:end);
end

disp(['Neural data extracted: ' num2str(toc) ' seconds'])

% Extract kinematic data
tic

kin_ts = downsample(kin.pos(:,1),round(1000*dt));  % This doesn't need to be filtered (it's just timestamps)

% Decimate (smooth+downsample, rather than just downsample)
x_pos = decimate(kin.pos(:,2),round(1000*dt)); % Sampling rate of kinematics is 1 kHz. thus, decimate so bin size is same as dt
y_pos = decimate(kin.pos(:,3),round(1000*dt));
x_vel = decimate(kin.vel(:,2),round(1000*dt));
y_vel = decimate(kin.vel(:,3),round(1000*dt));
x_acc = decimate(kin.acc(:,2),round(1000*dt));
y_acc = decimate(kin.acc(:,3),round(1000*dt));

disp(['Kinematic data extracted: ' num2str(toc) ' seconds'])

% Make timestamps
ts = edges(1:(end-1));
ts = ts(bin_start:end); % Discard first second as above

%% Arrange data by *trial*

clear Data; clear Data2; clear Data_trials

for block_idx = 1:length(blocks)
    block_num = blocks(block_idx);
    trials = find(bl_trials{block_num}==1);
    
    % Initialize
    num_trials = sum(bl_trials{block_num});
    Data(block_num).kinematics = cell(num_trials,1);
    Data(block_num).neural_data_M1 = cell(num_trials,1);
    Data(block_num).neural_data_PMd = cell(num_trials,1);
    Data(block_num).block_info = alldays(block_num).tt(trials,:);
    Data(block_num).trials = bl_trials{block_num};
    
    num_units_PMd = length(PMd_units);
    if M1_present
        num_units_M1 = length(M1_units);
    end
    
    % Arrange data by trial
    for trial_idx = 1:length(trials)
        trial_num = trials(trial_idx);
        
        disp(['Writing data for trial: ' num2str(trial_num)])
        
        tr_start = alldays(block_num).tt(trial_num,col_start_time) + time_before_extend; % Grab data before trial start
        tr_end = alldays(block_num).tt(trial_num,col_end_time) + time_end_extend; % Grab data after trial end
        tr_bins = logical((ts>=tr_start).*(ts<=tr_end)); % Find bins between tr_start and tr_end
        
        % Take neural data from between tr_start and tr_end
        Data(block_num).neural_data_PMd{trial_idx} = neural_data_temp_PMd(:,tr_bins);
        if M1_present
            Data(block_num).neural_data_M1{trial_idx} = neural_data_temp_M1(:,tr_bins);
        end
        
        % Take kinematics from between tr_start and tr_end
        Data(block_num).kinematics{trial_idx} = [x_pos(tr_bins), y_pos(tr_bins), ...
                                    x_vel(tr_bins), y_vel(tr_bins), ...
                                    x_acc(tr_bins), y_acc(tr_bins), ...
                                    kin_ts(tr_bins)];
                                
        % Get timestamps (imposed by me) to make sure they match those of
        % the kinematics
        Data(block_num).timestamps{trial_idx,1} = ts(tr_bins)';
        
    end
end

Data_trials = Data; % rename

%% Arrange data by *reach* - data has multiple reaches per trial; break trials up

% Note that the data extracted here can be further refined/windowed in
% visualize_data.m, and during any subsequent analysis scripts. 
% I.e., this isn't the final data. This data window is larger/more
% permissive.

% More user inputs
min_reach_len = 2; % in cm
time_before_cue = -0.3; % Amount of time before target comes on to grab data (in sec)
max_reach_time = round(1.4/dt) + ceil(abs(time_before_cue)/dt); % Max time for reach, in bins
spd_thresh = 8; % in cm/sec; This is different from the speed threshold in visualize_data.m; this helps to define the end of a reach
buff = 0.3; % Velocity often non-zero when cue comes on. Look forward at least this much to find end of reach (in sec)
end_buff = 0.3; % Allows reach end to be a little later than the official end of trial. Just to be a little more permissive (larger data window)
pd_lag = 0.096; % Photodetector wasn't used, so "cue on" is the command signal, not the detection signal. *Average* lag in Miller lab is 96 ms. Exact lag varies from trial to trial. See data description document for more information.

% Initialization
% Data2 = Data;
Data2 = struct;
idx = 1;

for tr = 1:num_trials
        reaches = find(~isnan(Data.block_info(tr,[3 8 13 18]))); % Find successful reaches in this trial. Unsuccessful reach will have a nan in corresponding column 3 (reach 1), 8 (2), 13 (3), 18 (4)
        num_reach = length(reaches);
        tr_end = Data.block_info(tr,col_end_time);

        ts = Data.kinematics{tr}(:,end); % Based upon kinematic time stamps

        x_vel = Data.kinematics{tr}(:,3);
        y_vel = Data.kinematics{tr}(:,4);
        x_pos = Data.kinematics{tr}(:,1);
        y_pos = Data.kinematics{tr}(:,2);
        
        spd = sqrt(x_vel.^2 + y_vel.^2);

        for reach_idx = 1:num_reach
            reach = reaches(reach_idx);

            idx_cue_on = 2 + 5*(reach - 1); % Find column in trial table which denotes time of target appearance
            
            % If reach wasn't completed or was invalid for some reason
            % (Columns referenced here will be NaN if reach is invalid)
            if isnan(Data.block_info(tr,idx_cue_on+1)) || isnan(Data.block_info(tr,idx_cue_on+2))
                continue
            end
            
            % Correct for command signal lag
            cue_on = Data.block_info(tr,idx_cue_on);
            cue_on = cue_on + pd_lag;
            
            % Correction: for reaches 2-4, target is displayed 100ms before
            % time in trial table
            if reach > 1
                cue_on = cue_on - .1; % Get the actual time from trial table; subtract 100 ms for correction
            end
            
            wind_st = cue_on + time_before_cue; % When the data window starts; grabs data before target appears

            [~,idx_cue_on2] = min(abs(ts-cue_on)); % Find time bin when cue comes on

            % Determine end time for each reach
            if reach < 4 % For reaches 1-3 on a given trial; these will be followed by another reach
                % Find time of: min velocity before next go cue
                idx_cue_on_next = 2 + 5*(reach);
                cue_on_next = Data.block_info(tr,idx_cue_on_next);
                
                % For reach to end: (this is permissive) 
                % 1) slow (falls below speed threshold), and
                % 2) a certain minimum time after cue onset must have elapsed (buff) 
                % 3) end has to be before the next reach starts plus buff
                cond_reach_end = logical((spd < spd_thresh).*(ts > (cue_on + buff)).*(ts < (cue_on_next + buff)));
                
                % If conditions not met, skip
                if sum(cond_reach_end) == 0
                    continue
                end
                
                % Within times meeting conditions, find one with minimum speed                
                cond_reach_end_nan = 1.*cond_reach_end; % Initialize and convert to scalar array
                cond_reach_end_nan(cond_reach_end_nan < 1) = nan; % Make the bins not meeting conditions above nan
                [~,idx_reach_end] = min(spd.*cond_reach_end_nan); 
                reach_end = ts(idx_reach_end);
                
                
            else
                % Find time of min velocity after last reach. Conditions:
                % 1) slow (falls below speed threshold), and
                % 2) a certain minimum time after cue onset must have elapsed (buff) 
                % 3) end has to be before the trial ends plus buff
                cond_reach_end = logical((spd < spd_thresh).*(ts > (cue_on + buff)).*(ts < (tr_end + end_buff))); 
                
                % If conditions not met, skip
                if sum(cond_reach_end) == 0
                    continue
                end
                
                % Within times meeting conditions, find one with minimum speed                
                cond_reach_end_nan = 1.*cond_reach_end; % initialize and convert to scalar array
                cond_reach_end_nan(cond_reach_end_nan < 1) = nan;
                [~,idx_reach_end] = min(spd.*cond_reach_end_nan);
                reach_end = ts(idx_reach_end);

            end

            
            % Define window of time to save
            wind_reach = logical((ts>=wind_st).*(ts<=reach_end));
            
            
            % Add meta-data
            Data2.trial_num{idx,1} = tr;
            Data2.reach_num{idx,1} = reach;
            Data2.reach_st{idx,1} = cue_on; % Time of cue on used as a proxy for when reach starts. It's approximate.
            Data2.cue_on{idx,1} = cue_on;
            Data2.reach_end{idx,1} = reach_end;
            Data2.reach_pos_st{idx,1} = Data.kinematics{tr}(idx_cue_on2,1:2);
            Data2.reach_pos_end{idx,1} = Data.kinematics{tr}(idx_reach_end,1:2);
            delta_pos = Data2.reach_pos_end{idx,1} - Data2.reach_pos_st{idx,1};
            [Data2.reach_dir{idx,1}, Data2.reach_len{idx,1}] = cart2pol(delta_pos(1),delta_pos(2));
            
            idx_target_on = 1 + ceil(abs(time_before_cue)/dt); % Target is on in bin 1 unless extra time before is added, in which case it's on after that extra time
            temp = zeros(sum(wind_reach),1); temp(idx_target_on) = 1;
            Data2.target_on{idx,1} = temp; 

            % Copy stuff
            Data2.kinematics{idx,1} = Data.kinematics{tr}(wind_reach,:);
            Data2.neural_data_PMd{idx,1} = Data.neural_data_PMd{tr}(:,wind_reach);
            if M1_present
                Data2.neural_data_M1{idx,1} = Data.neural_data_M1{tr}(:,wind_reach);
            end
            Data2.block_info = Data.block_info;
            Data2.time_window{idx,1} = wind_reach;
            Data2.timestamps{idx,1} = Data.timestamps{tr}(wind_reach);

            % Exclude reach if it doesn't meet requirements
            if (Data2.reach_len{idx} < min_reach_len) || (sum(wind_reach) > max_reach_time)
                continue
            end

            % Increment reach index
            idx = idx + 1;

        end
end

% Rename variables
clear Data
Data = Data2; clear Data2

disp('Data cleaning completed.')
disp('Reach data located in variable: Data.')
disp('Trial data located in variable: Data_trials.')