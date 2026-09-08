%simulate data and run PDAT
D = 60;
Ntrain = 200;
Ntest = 200;
T = 100;
p = uncertainty_eeg_params;
K = p.nchan;

signallevel = 1; 
noiselevel = 10;

%% Simulate Data 
%W will start at 0's (no signal) and approach this logarithmically over
%time
trueW = randn(D,K)*signallevel;
Wt = zeros(D,K,T);
tvec = log(linspace(1, exp(3), T))/3;
for i = 1:numel(trueW)
    [a,b] = ind2sub([D,K],i);
    Wt(a,b,:) = trueW(i)*tvec;
end

%cov will be static
dfcov = 5; truenoise = randn(D,dfcov);
truecov = (truenoise*truenoise')/dfcov;
truecov(eye(D)==1) = truecov(eye(D)==1) + rand(D,1);

%stimuli in train and test evenly tile the circle
train.stimval = linspace(0, 2*pi, Ntrain+1)';
train.stimval(end) = [];
test.stimval = linspace(0, 2*pi, Ntest+1)'; test.stimval(end)=[];

basis_prefs = (0:2*pi/(p.nchan):2*pi);
basis_prefs(end) = [];

%compute basis tuning function responses to discretized stimulus values
p.basis_resp = nan(p.nbinsstimval, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        p.basis_resp(:,j,i) = max(0,cos(p.binvals - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^p.chan_exp);
    end
end

%compute basis tuning function responses to stimulus values of training set
train.resp = nan(Ntrain, K,p.nsets);
for i = 1:p.nsets
    for j = 1:K
        train.resp(:,j,i) = max(0,cos(train.stimval - (basis_prefs(j)+((i-1)*(2*pi/K/p.nsets)))).^p.chan_exp);
    end
end
%and for testing set
test.resp = nan(Ntest, K,p.nsets);
for i = 1:p.nsets
    for j = 1:K
        test.resp(:,j,i) = max(0,cos(test.stimval - (basis_prefs(j)+((i-1)*(2*pi/K/p.nsets)))).^p.chan_exp);
    end
end

%simulate data
train.allsamples = zeros(T,D,Ntrain);
for t = 1:T
    w=squeeze(Wt(:,:,t));
    for i = 1:Ntrain
        signal = squeeze(train.resp(i,:,1))*w';
        noise = mvnrnd(zeros(1,D), truecov);
        train.allsamples(t,:,i) = signal+noise;
    end
end
test.allsamples = zeros(T,D,Ntest);
for t = 1:T
    w=squeeze(Wt(:,:,t));
    for i = 1:Ntest
        signal = squeeze(test.resp(i,:,1))*w';
        noise = mvnrnd(zeros(1,D), truecov)*noiselevel;
        test.allsamples(t,:,i) = signal+noise;
    end
end

%% Run PDAT
errmat = zeros(Ntest, T);
uncmat = zeros(Ntest, T);
corrmat = zeros(1,T);

train.boot.idx = zeros(Ntrain, p.nboot);
for b = 1:p.nboot
    train.boot.idx(:,b) = randsample(1:Ntrain, Ntrain, true);
end
train.boot.sets = randsample(1:p.nsets, p.nboot, true);

train.cvid = repmat([1:p.cvk]', ceil(Ntrain/p.cvk), 1);
train.cvid = train.cvid(1:Ntrain);
train.cvid = train.cvid(randperm(size(train.cvid, 1)));

for t = 1:T
    [outp, uncertainty] = TAFKAP_decode(t,train,test,p);
    errmat(:,t) = circ_dist(outp, test.stimval);
    uncmat(:,t) = uncertainty;
    corrmat(t) = corr(abs(errmat(:,t)), uncertainty);
end

%% show results

figure
plot(mean(abs(errmat),1));
xlabel('Time'); ylabel('Mean absolute decoding error');

figure
plot(mean(uncmat,1));
xlabel('Time'); ylabel('Mean decoded uncertainty');

figure
plot(corrmat);
xlabel('Time'); ylabel({'Trial-by-trial correlation between decoding', 'error and decoded uncertainty'});
yline(0);
