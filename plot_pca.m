function plot_pca()

    data_files = [
        "source_data/firing_rates/MM_S1_fr.mat";
        "source_data/firing_rates/MT_S1_fr.mat";
        "source_data/firing_rates/MT_S2_fr.mat";
        "source_data/firing_rates/MT_S3_fr.mat";
    ];

    fr_session = load(data_files(1));
    M1_population = fr_session.session.Data.neural_data_M1;
    PMd_population = fr_session.session.Data.neural_data_PMd;
    timestamps = fr_session.session.Data.timestamps;
    start_time = fr_session.session.Data.reach_st;
    end_time = fr_session.session.Data.reach_end;
    trial_num = fr_session.session.Data.trial_num;
    reach_order = fr_session.session.Data.reach_num;
    cue_on = fr_session.session.Data.cue_on;
    reach_num = 40;

    [M1_coeff, M1_coeff_score, M1_coeff_latent, M1_coeff_tsquared, M1_coeff_explained, M1_coeff_mu] = pca(M1_population{reach_num}.');
    [PMd_coeff, PMd_coeff_score, PMd_coeff_latent, PMd_coeff_tsquared, PMd_coeff_explained, PMd_coeff_mu] = pca(PMd_population{reach_num}.');

    pc1_M1 = M1_coeff_score(:, 1);
    pc2_M1 = M1_coeff_score(:, 2);
    pc3_M1 = M1_coeff_score(:, 3);

    pc1_PMd = PMd_coeff_score(:, 1);
    pc2_PMd = PMd_coeff_score(:, 2);
    pc3_PMd = PMd_coeff_score(:, 3);

    figure
    plot3(pc1_M1, pc2_M1, pc3_M1)
    hold on;
    plot3(pc1_PMd, pc2_PMd, pc3_PMd)
    hold off;

    figure
    plot(timestamps{reach_num}, pc1_M1, "Color", "red")
    hold on;
    plot(timestamps{reach_num}, pc1_PMd, "Color", "blue")
    hold on;
    xline(start_time{reach_num}, '--w');
    hold on;
    xline(end_time{reach_num}, '--w')
    hold on;
    xline(cue_on{reach_num}, '--m')
    disp(trial_num{reach_num})
    disp(reach_order{reach_num})
    disp(size(timestamps{reach_num}))





end