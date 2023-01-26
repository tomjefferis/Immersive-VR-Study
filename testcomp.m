% load data
ch1 = readtable('data/EEG device 2/EEGCh1_EA376645F366_11.07.46_250.csv');
ch2 = readtable('data/EEG device 2/EEGCh2_EA376645F366_11.07.46_250.csv');
ch3 = readtable('data/EEG device 2/EEGCh3_EA376645F366_11.07.46_250.csv');
ch4 = readtable('data/EEG device 2/EEGCh4_EA376645F366_11.07.46_250.csv');
fs = 250;

%drop column 1
dat1 = table2array(ch1(fs*90:fs*300,2:end));
dat2 = table2array(ch2(fs*90:fs*300,2:end));
dat3 = table2array(ch3(fs*90:fs*300,2:end));
dat4 = table2array(ch4(fs*90:fs*300,2:end));
x = dat1(1:end-1) - dat3(1:end-1);

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