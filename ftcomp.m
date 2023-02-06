addpath('W:\PhD\MatlabPlugins\fieldtrip-20210906'); % path to fieldtrip
addpath('W:\PhD\MatlabPlugins\spm12') % path to spm

ft_defaults;

% load data

list = dir('../EEG/*.fif');
data = {};
scores = {};
%if freq data doesnt exist, create it
for i = 1:length(list)
    file = list(i).name;
    splitFile = split(file, '_');
    participantNumber = str2double(splitFile{1});

    if ~endsWith(file, 'resting.fif')

        if i ~= 1
            sclen = size(scores, 2);
        else
            sclen = 0;
        end

        if (contains(file, 'watching') || contains(file, 'watch'))
            scores{1, sclen + 1} = 1;
            scores{2, sclen + 1} = participantNumber;
        elseif (contains(file, 'normal') || contains(file, 'correct'))
            scores{1, sclen + 1} = 2;
            scores{2, sclen + 1} = participantNumber;
        elseif (contains(file, 'hard'))
            scores{1, sclen + 1} = 3;
            scores{2, sclen + 1} = participantNumber;
        end

        filepath = ['../EEG/' file];
        fprintf('%s\n', file);
        cfg = [];
        cfg.dataset = filepath;
        % append preprocessed data to list data
        temp = ft_preprocessing(cfg);
        cfg = [];
        cfg.resamplefs = 100;
        temp = ft_resampledata(cfg, temp);
        cfg = [];
        cfg.output = 'pow';
        cfg.method = 'wavelet';
        cfg.width = 5;
        cfg.foi = 5:1:35;
        % toi from 60 to 500 seconds interval of 1/256
        cfg.toi = 60:(1/100):500;
        temp = ft_freqanalysis(cfg, temp);
        temp = temp.powspctrm;
        % save temp to file with participant number and condition
        tt = split(splitFile{2},'.');
        save(['../FreqData/' num2str(participantNumber) '_' tt{1},'.mat'], 'temp');
    end

end
