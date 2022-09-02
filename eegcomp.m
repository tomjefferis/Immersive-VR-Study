
filename = '10_normal.bdf';

%addpath('W:\PhD\MatlabPlugins\fieldtrip-20210906'); % path to fieldtrip
cfg = [];
cfg.datafile = filename;
cfg.headerfile = filename;
cfg.reref = 'yes';
cfg.channel = {'all','-A1','-A2','-LEOG','-REOG','-UEOG','-DEOG','-EXG7','-EXG8','-GSR1','-GSR2','-Erg1','-Erg2','-Resp','-Plet','-Temp','-Status'};
cfg.refchannel = 'all';
cfg.dftfilter = 'yes';
cfg.dtffreq = [50 100 150];
cfg.bpfilter = 'yes';
cfg.bpfreq = [1 60];
cfg.chantype = 'eeg';
cfg.continuous = 'yes';
%cfg.baselinewindow = [-0.2 0];
data = ft_preprocessing(cfg);

cfg = [];
cfg.resamplefs = 250;
data = ft_resampledata(cfg, data);

fs = 250;      
x = data.trial{1}(17,:);
%t = x(:,1);
%x = x(:,2);

%x = x - mean(x);
%x = bpfilt(x,0.1,30,fs);
%pspectrum(x,250);
y = fft(x);
figure;
n = length(x);          % number of samples
f = (0:n-1)*(fs/n);     % frequency range
power = abs(y).^2/n;    % power of the DFT
pspectrum(x,250)
plot(f,power)
xlabel('Frequency')
ylabel('Power')
xlim([0 30])
%ylim([0 0.002])