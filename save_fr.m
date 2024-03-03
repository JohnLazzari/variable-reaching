function save_fr()

    total_units_M1 = [67, 0, 0, 0];
    total_units_PMd = [94, 49, 46, 57];

    data_files = [
        "source_data/processed/MM_S1_processed.mat";
        "source_data/processed/MT_S1_processed.mat";
        "source_data/processed/MT_S2_processed.mat";
        "source_data/processed/MT_S3_processed.mat";
    ];

    save_names = [
        "source_data/firing_rates/MM_S1_";
        "source_data/firing_rates/MT_S1_";
        "source_data/firing_rates/MT_S2_";
        "source_data/firing_rates/MT_S3_";
    ];

    num_sessions = 4;

    for i = 1:num_sessions

        num_units_M1 = total_units_M1(i);
        num_units_PMd = total_units_PMd(i);
        fileName = convertStringsToChars(data_files(i));
        session = load(fileName);

        %% Go through M1 neurons
        for neuron = 1:num_units_M1

            for reach = 1:size(session.Data.reach_num, 1)

                % Saving trial/reach averaged activity
                cur_spike_times = session.Data.neural_data_M1{reach}(neuron, :) / 0.01;
                filtered_spike_times = smooth(cur_spike_times, 10);
                session.Data.neural_data_M1{reach}(neuron, :) = filtered_spike_times.';
            
            end

        end


        %% Go through PMd neurons
        for neuron = 1:num_units_PMd

            for reach = 1:size(session.Data.reach_num, 1)

                % Not implementing different conditions yet
                % Saving trial/reach averaged activity
                cur_spike_times = session.Data.neural_data_PMd{reach}(neuron, :) / 0.01;
                filtered_spike_times = smooth(cur_spike_times, 10);
                session.Data.neural_data_PMd{reach}(neuron, :) = filtered_spike_times.';
            
            end

        end

        disp(i)
        kinematics = session.Data.kinematics;
        timestamps = session.Data.timestamps;
        start_time = session.Data.reach_st;
        end_time = session.Data.reach_end;
        trial_num = session.Data.trial_num;
        reach_order = session.Data.reach_num;
        cue_on = session.Data.cue_on;
        reach_dir = session.Data.reach_dir;
        target_on = session.Data.target_on;
        PMd_population = session.Data.neural_data_PMd;
        
        save(save_names(i) + "PMd_fr.mat", "PMd_population")
        save(save_names(i) + "cue_on.mat", "cue_on")
        save(save_names(i) + "reach_order.mat", "reach_order")
        save(save_names(i) + "trial_num.mat", "trial_num")
        save(save_names(i) + "end_time.mat", "end_time")
        save(save_names(i) + "start_time.mat", "start_time")
        save(save_names(i) + "timestamps.mat", "timestamps")
        save(save_names(i) + "kinematics.mat", "kinematics")
        save(save_names(i) + "reach_dir.mat", "reach_dir")
        save(save_names(i) + "target_on.mat", "target_on")

    end

end

function out = filterGauss(in, SD)

    % usage: out = filterGauss(in, SD);
    %
    % Filters a spiketrain (ones and zeros, one element per ms) with a gaussian.
    % The output is rate (in spikes per ms).
    % To convert to spikes/s, multiply by 1000.
    %
    % For convenience, the output is the same length as the input, but of 
    % course the very early and late values are not 'trustworthy'.
    %
    % same filtering as in 'FilterSpikes' except the output is the same orientation as the input
    % also more robust if SD is not an integer
    % the input arguments are reversed relative to FilterSpikes
    % whatever orientation it cam in, make it a row vector
    s = size(in);
    
    if s(1) > 1 && s(2) > 1
        disp('You sent in an array rather than a vector');
        return
    end
    
    if s(1) > 1   % if it came in as a column vector
        in = in';  % make it a ow vector
        flip = false;  % and remember to flip back at the very end
    else
    %    flip = false;
        flip = false;
    end
    % compute the normalized gaussian kernel
    SDrounded = 2 * ceil(SD/2);  % often we just need SD to set the number of points we keep and so forth, so it needs to be an integer.
    gausswidth = 12*SDrounded;  
    F = normpdf(1:gausswidth, gausswidth/2, SD);
    F = F/(sum(F));
    
    % pad, filter
    shift = floor(length(F)/2); % this is the amount of time that must be added to the beginning and end;
    last = length(in); % length of incoming data
    prefilt = [zeros(1,shift)+mean(in(1:SDrounded)), in, (zeros(1,shift)+mean(in(last-SDrounded:last)))]; % pads the beginning and end with copies of first and last value (not zeros)
    postfilt = filter(F,1,prefilt); % filters the data with the impulse response in Filter
    
    % trim, flip orientation if necessary, and send out
    if ~flip  % if it came in as a row vector
        out = postfilt(2*shift:length(postfilt)-1);  % Shifts the data back by 'shift', half the filter length
    else
        out = postfilt(2*shift:length(postfilt)-1)';  % if it was a column vector flip back.
    end
end
