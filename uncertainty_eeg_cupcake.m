function [errmat, postmat, uncmat, corrmat, covparams, timestr, trainset, testset, p] = uncertainty_eeg_cupcake(p)

if nargin<1
    p = uncertainty_eeg_params;
end

warning('off', 'MATLAB:rankDeficientMatrix');

eeg = load(p.eeg_file);
trials = load(p.trial_file);
samplerate = eeg.outp.hz;
timewindow = eeg.outp.time>=p.timewindow(1)&eeg.outp.time<=p.timewindow(2);

%% EEG preprocess%%
%for cv estimation of optimal shrinkage parameter
% cvid = Shuffle(repmat([1:p.cvk]', ceil(fullN/p.cvk), 1));
% want approximately equal trials of each orientation in each cv fold
%chan x time x trialh
%want time x chan x trial
targetchannels = getchans(eeg.outp.chanlocs, p.targetchannels);
if isfield(eeg.outp, 'excludechannels')
    targetchannels = setdiff(targetchannels, eeg.outp.excludechannels);
    if p.rmInterp
        targetchannels = setdiff(targetchannels, eeg.outp.interpchans);
    end
end
Nchan = size(targetchannels,2);

if p.dobaseline
    blwindow = find(ismember(eeg.outp.time, [-p.baselinetimewindow:0]));
    eeg.outp.data = bsxfun(@minus, eeg.outp.data, mean(eeg.outp.data(:,blwindow,:),2));
end

if any(p.freq_range) %do this before extracting time window
    addpath(genpath('/projectnb/rdenlab/Tools/eeglab2023.0'));
    temp = eeg.outp.data(targetchannels,:,:);
    temp = reshape(temp, Nchan, []);
    temp = eegfilt(double(temp), samplerate, p.freq_range(1), p.freq_range(2), size(eeg.outp.data,2));
    temp = reshape(temp, Nchan, size(eeg.outp.data,2), []);
    allsamples = permute(temp, [2,1,3]);
    allsamples = abs(hilbert(allsamples)).^2;
    allsamples = allsamples(timewindow,:,:);
    %do you ever baseline the power-transformed data?
else
    allsamples = double(eeg.outp.data(targetchannels,timewindow,:));
    allsamples = permute(allsamples, [2,1,3]);
end

Ntrials = size(allsamples, 3);


stimval = trials.expt.trialsPresented.(p.feature)';
iscue = logical(trials.expt.trialsPresented.att);

removeTrials = false(Ntrials,1);

if isfield(eeg.outp, 'excludetrials')

    if ~isempty(p.eyecutoff)
        removeTrials(eeg.outp.excludetrials{p.eyecutoff}) = true;
    else
        removeTrials(eeg.outp.excludetrials) = true;
    end
end

switch p.train_test
    case 'precue'
        removeTrials(trials.expt.trialsPresented.att==0) = 1;
    case 'nocue'
        removeTrials(trials.expt.trialsPresented.att==1) = 1;
end

if p.featurebin_subsamp %this and featurebin should be done after selecting precue/nocue trials
    if p.iscircular
        edges = 0:(2*pi/p.featurebin_subsamp):(2*pi);
    else
        edges = linspace(min(stimval), max(stimval),p.featurebin_subsamp+1);
    end
    remvec = 1:numel(removeTrials);
    remvec(removeTrials) = [];
    binid = arrayfun(@(x) find(x<=edges, 1), stimval(~removeTrials))-1;
    numperbin = groupcounts(binid);
    minbin = min(numperbin);
    binremove = arrayfun(@(x) randsample(remvec(binid==x), numperbin(x)-minbin), 1:p.featurebin_subsamp, 'UniformOutput', false);
    if p.verbose
        fprintf('\nremoved %i trials in order to attain equally populated feature bins', numel(cell2mat(binremove)));
    end
    removeTrials(cell2mat(binremove)) = true;
end

allsamples(:,:,removeTrials) = [];
stimval(removeTrials) = [];
iscue(removeTrials) = [];
Ntrials = Ntrials - sum(removeTrials);

if p.featurebins
    stimval = discretize(stimval, quantile(stimval, linspace(0,1, p.featurebins+1)));
end

if p.slidingwindow
    NtimePoints = size(allsamples,1)-p.msperbin+1;
    x = allsamples;
    allsamples = zeros(NtimePoints, Nchan, Ntrials);
    for i = 1:NtimePoints
        allsamples(i,:,:) = mean(x(i:(i+p.msperbin-1),:,:),1);
    end
else
    %time binning
    % cut off data at end if trial length is not divisible by binsize
    NtimePoints = floor(size(allsamples,1)/p.msperbin);
    allsamples = allsamples(1:NtimePoints*p.msperbin,:,:);
    allsamples = reshape(mean(reshape(allsamples,p.msperbin,[]),1),NtimePoints,Nchan,Ntrials);
end

% if p.pseudotrials %not currently implemented
%     [allsamples, stimval, contrast] = make_pseudotrials(allsamples, stimval, contrast);
%     Ntrials = size(allsamples,3);
% end

if p.dosensorzscore
    means = mean(allsamples,3);
    sds = std(allsamples,[],2);
    allsamples = bsxfun(@minus,allsamples,means);
    allsamples =bsxfun(@rdivide, allsamples, sds);
end

if p.dopatternzscore
    means = mean(allsamples,2);
    sds = std(allsamples, [], 2);
    allsamples = bsxfun(@minus, allsamples, means);
    allsamples = bsxfun(@rdivide, allsamples, sds);
end

%% Channel response precompute %%
p.binvals = linspace(0, 2*pi, p.nbinsstimval+1)';
p.binvals(end) = [];

basis_prefs = (0:2*pi/(p.nchan):2*pi);
basis_prefs(end) = [];
p.basis_resp = nan(p.nbinsstimval, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        p.basis_resp(:,j,i) = max(0,cos(p.binvals - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^p.chan_exp);
    end
end

resp = nan(Ntrials, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        resp(:,j,i) = max(0,cos(stimval - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^p.chan_exp);
        %         resp(:,j,i) = resp(:,j,i) - mean(resp(:,j,i));
    end
end

if p.normchan
    p.basis_resp = bsxfun(@rdivide, p.basis_resp, sum(p.basis_resp,2));
    resp = bsxfun(@rdivide, resp, sum(resp,2));
end

%The basis responses may also be demeaned s.t. their average over
%trials equals zero for each basis channel

%for making figures
timestr = strings(NtimePoints,1);
if p.msperbin==1
    for m = 1:NtimePoints
        timestr(m) = sprintf('t%d', eeg.outp.time(timewindow(m)));
    end
elseif p.slidingwindow == 1
    tmat = eeg.outp.time(timewindow);
    for m = 1:NtimePoints
        timestr(m) = sprintf('t%d - t%d', tmat(m), tmat(m)+p.msperbin-1);
    end
else
    tmat = eeg.outp.time(timewindow);
    for m = 1:NtimePoints
        timestr(m) = sprintf('t%d - t%d', tmat((m-1)*p.msperbin+1), tmat(m*p.msperbin));
    end
end

%~p.loocv: single run with a test_part % split
% ## make an option to run each fold as test
if ~p.loocv
    Ntesttrials = floor(p.test_part*Ntrials/2)*2;
    Ntraintrials = Ntrials-Ntesttrials;
    if p.featurebin_subsamp %also make sure test trials and cv folds have as close to uniform as possible distribution of stimvals

        vec = 1:Ntrials;
        binid = arrayfun(@(x) find(x<=edges, 1), stimval)-1;
        ttestperbin = repmat(floor(Ntesttrials/p.featurebin_subsamp), [1,p.featurebin_subsamp]);
        xx = cell2mat(arrayfun(@(x) datasample(vec(binid==x), ttestperbin(x), 'Replace', false), unique(binid)', 'UniformOutput', false));
        vec(xx) = []; binid(xx) = [];
        bintosample = 0;
        while numel(xx)<Ntesttrials
            bintosample = mod(bintosample, p.featurebin_subsamp)+1;
            smpl = datasample(vec(binid==bintosample),1, 'Replace', false);
            xx = [xx smpl]; binid(vec==smpl) = []; vec(vec==smpl) = [];
        end
        test.trials = xx;

        trialsperfold = repmat(floor(Ntraintrials/p.cvk), [1,p.cvk]);
        trialsperfold(1:(Ntraintrials-sum(trialsperfold))) = ceil(Ntraintrials/p.cvk);

        train.trials = [];
        for j = 1:numel(trialsperfold)
            tnum = floor(trialsperfold(j)/p.featurebin_subsamp);
            tids = arrayfun(@(x) datasample(vec(binid==x), tnum, 'Replace', false), 1:p.featurebin_subsamp, 'UniformOutput', false);
            xx = cell2mat(tids);
            while numel(xx)<trialsperfold(j)
                bintosample = mod(bintosample, p.featurebin_subsamp)+1;
                smpl = datasample(vec(binid==bintosample),1, 'Replace', false);
                xx = [xx smpl]; binid(vec==smpl) = []; vec(vec==smpl) = [];
            end
            train.trials = [train.trials xx];
        end

        if p.nboot~=1
            binids = arrayfun(@(x) find(x<=edges, 1), stimval(train.trials))-1;
            sm = arrayfun(@(x) sum(binids==x), 1:p.featurebin_subsamp);
            cs = [0 cumsum(sm)];
            vec = 1:Ntraintrials;
            for j = 1:p.featurebin_subsamp
                boots = randsample(vec(binids==j), sm(j)*p.nboot, true);
                train.boot.idx((cs(j)+1):cs(j+1),:) = reshape(boots, [sm(j), p.nboot]);
            end
        else
            train.boot.idx = [1:Ntraintrials]';
        end
        train.cvid = cell2mat(arrayfun(@(x) repmat(x, [1,trialsperfold(x)]), 1:p.cvk, 'UniformOutput', false));
    else
        Ntraintrials = Ntrials-Ntesttrials;
        shuff = 1:Ntrials;
        test.trials = randsample(shuff, Ntesttrials);
        train.trials = setdiff(shuff, test.trials);

        if ~isempty(p.ntraintrials)
            Ntraintrials = p.ntraintrials;
            train.trials = randsample(train.trials, Ntraintrials);
        end
        if ~isempty(p.train_subset)
            Ntraintrials = ceil(Ntraintrials*p.train_subset);
            train.trials = randsample(train.trials, Ntraintrials);
        end

        train.cvid = repmat([1:p.cvk]', ceil(Ntraintrials/p.cvk), 1);
        train.cvid = train.cvid(1:Ntraintrials);
        train.cvid = train.cvid(randperm(size(train.cvid, 1)));

        if p.nboot~=1
            train.boot.idx = randsample(Ntraintrials, Ntraintrials*p.nboot, true);
            train.boot.idx = reshape(train.boot.idx, [Ntraintrials, p.nboot]);
        else
            train.boot.idx = (1:Ntraintrials)';
        end

    end
    train.boot.sets = randi(p.nsets, [1, p.nboot]);

    train.resp = resp(train.trials,:,:);
    test.resp = resp(test.trials,:,:);
    train.allsamples = allsamples(:,:,train.trials);
    test.allsamples = allsamples(:,:,test.trials);
    train.stimval = stimval(train.trials);
    test.stimval = stimval(test.trials);
    if p.gennull
        train.perm = randperm(numel(train.trials));
        train.stimval = train.stimval(train.perm);
        train.resp = train.resp(train.perm,:,:);
    end
    %this part figures out the index of the trials in train and test within
    %the original dataset, before excluding trials
    nremain = cumsum(~removeTrials);
    trainset = zeros(1, Ntraintrials);
    for i = 1:Ntraintrials
        trainset(i) = find(nremain==train.trials(i), 1,'first');
    end
    testset = zeros(1, Ntesttrials);
    for i = 1:Ntesttrials
        testset(i) = find(nremain==test.trials(i), 1, 'first');
    end

    %empirical covariance (not noise covariance) of whole timeseries/ of
    %pre-stimulus epoch
    if p.covmethod=="signal"
        x = reshape(permute(train.allsamples, [1 3 2]), [size(train.allsamples,1)*size(train.allsamples,3), size(train.allsamples,2)]);
        x = x-repmat(mean(x,2),1,size(x,2));
        train.cov = x'*x/size(x,1);
    end
    if p.covmethod=="baseline"
        y = train.allsamples(1:floor(p.baselinetimewindow/p.msperbin));
        x = reshape(permute(y, [1 3 2]), [size(y,1)*size(y,3), size(y,2)]);
        x = x-repmat(mean(x,2),1,size(x,2));
        train.cov = x'*x/size(x,1);
    end
    if p.covmethod == "FA"
        train.bestk = zeros(1,NtimePoints);
    end

    %initialize summary stat arrays
    postmat = zeros(Ntesttrials, p.nbinsstimval, NtimePoints,p.outputformat);
    uncmat = zeros(NtimePoints,Ntesttrials,p.outputformat);
    errmat = zeros(NtimePoints,Ntesttrials,p.outputformat);
    corrmat = zeros(NtimePoints,1,p.outputformat);

    if p.crosstime
        postmat = repmat(postmat, [1,1,1,NtimePoints]); %trial x stimvalbin x traintime x testtime
        uncmat = repmat(uncmat, [1,1,NtimePoints]); %testtime x trial x traintime
        errmat = repmat(errmat, [1,1,NtimePoints]); %testtime x trial x traintime
        corrmat = repmat(corrmat, [1, NtimePoints]); %testtime x traintime
    end

    if p.pca
        test.allsamples= zeros(NtimePoints, p.npcs, Ntesttrials);
        train.allsamples= zeros(NtimePoints, p.npcs, Ntraintrials);
    end
    if strcmp(p.regressionmethod, 'FLS')
        fitweights(train.allsamples,train.resp,[], [], p, train.boot); %no output, it saves a persistent
    end

    if p.verbose; fprintf('\n LOOP START: %i iterations', NtimePoints*(NtimePoints^p.crosstime)); end
    for m = 1:NtimePoints

        if p.pca
            coeff = pca(squeeze(allsamples(m,:,train.trials))', 'NumComponents', p.npcs);
            temp_samples = squeeze(allsamples(m,:,:))'*coeff;
            temp_samples = permute(temp_samples, [3,2,1]);
            train.allsamples(m,:,:) = temp_samples(:,:,train.trials);
            test.allsamples(m,:,:) = temp_samples(:,:,test.trials);
        end

        %ordinary timepoint-by-timepoint case
        if ~p.crosstime
            if p.regressionmethod == "SVM"
                outp = SVM_decode(m,train,test,p);
                if p.iscircular
                    errors = circ_dist(outp', test.stimval);
                    avgerrort = mean(abs(errors))/(2*pi);
                else
                    errors = test.stimval - outp';
                    avgerrort = mean(abs(errors));
                end
                errmat(m,:) = errors;
                if p.verbose
                    fprintf('\n t=%s, error=%3.2f', timestr(m), avgerrort);
                end

            else
                [outp, uncertainty, posteriors, covp] = TAFKAP_decode(m,train,test,p);

                errors = circ_dist(outp, test.stimval);
                avgerrort = mean(abs(errors))/(2*pi);
                avgunct = mean(uncertainty);
                errmat(m,:) = errors;
                uncmat(m,:) = uncertainty;
                if p.verbose
                    fprintf('\n t=%s, error=%3.2f, uncertainty=%3.2f', timestr(m), avgerrort, avgunct);
                end
                postmat(:,:,m) = posteriors;
                corrmat(m) = corr(abs(errors),uncertainty);
                covparams(m)=covp;
            end
            %temporal generalization case
        else
            for w=1:NtimePoints
                if p.regressionmethod == "SVM"
                    outp = SVM_decode(m,train,test,p,w);
                    if p.iscircular
                        errors = circ_dist(outp', test.stimval);
                        avgerrort = mean(abs(errors))/(2*pi);
                    else
                        errors = test.stimval - outp';
                        avgerrort = mean(abs(errors));
                    end
                    errmat(m,:,w) = errors;
                    if p.verbose
                        fprintf('\n t=%s, error=%3.2f', timestr(m), avgerrort);
                    end

                else
                    [outp, uncertainty, posteriors, covp] = TAFKAP_decode(m,train,test,p,w);
                    errors = circ_dist(outp, test.stimval);
                    avgerrort = mean(abs(errors))/(2*pi);
                    avgunct = mean(uncertainty);
                    errmat(m,:,w) = errors;
                    uncmat(m,:,w) = uncertainty;
                    if p.verbose
                        fprintf('\n train:t=%s, test:t=%s, error=%3.2f, uncertainty=%3.2f', timestr(m), timestr(w), avgerrort, avgunct);
                    end
                    postmat(:,:,m,w) = posteriors;
                    corrmat(m,w) = corr(abs(errors),uncertainty);
                    covparams(m,w) = covp;
                end
            end
        end
    end

    %save this in p just because
    if p.covmethod == "EM"
        p.klikelihood = klikelihoodcell;
    end

    %leave-one-out cross-validation case (ntrials times slower or more)
else
    trial_select = 1:Ntrials;

    nremain = cumsum(~removeTrials);
    trainset = zeros(1, Ntrials);
    trialvec = 1:Ntrials;
    for i = 1:Ntrials
        trainset(i) = find(nremain==trialvec(i), 1,'first');
    end

    %     switch p.train_test
    %         case 'precue'
    %             trial_select = trial_select(iscue);
    %         case 'nocue'
    %             trial_select =  trial_select(~iscue);
    %     end

    Ntrials = numel(trial_select);

    if ~isempty(p.ntraintrials)
        Ntrials = p.ntraintrials;
        trial_select = randsample(trial_select, Ntrials);
    end
    if ~isempty(p.train_subset)
        Ntrials = ceil(Ntrials*p.train_subset);
        trial_select = randsample(trial_select, Ntrials);
    end

    trainset = trainset(trial_select);
    testset = trainset;

    allsamples = allsamples(:,:,trial_select);
    stimval = stimval(trial_select);
    resp = resp(trial_select, :, :);

    postmat = zeros(Ntrials, p.nbinsstimval, NtimePoints);
    uncmat = zeros(NtimePoints,Ntrials);
    errmat = zeros(NtimePoints,Ntrials);
    corrmat_time = zeros(Ntrials);
    corrmat_trials = zeros(NtimePoints);

    if p.crosstime
        %don't save posteriors or gamma it's too big
        postmat = [];
        corrmat = [];

        uncmat = repmat(uncmat, [1,1,NtimePoints]);
        errmat = repmat(errmat, [1,1,NtimePoints]);
    end

    if p.verbose; fprintf('\n LOOP START: %i iterations', Ntrials*NtimePoints*(NtimePoints^(p.crosstime))); end
    for trial = 1:Ntrials
        test = struct();
        train = struct();
        trialtime = tic;
        test.trials = trial;

        NtestTrials = numel(test.trials);
        Ntraintrials = Ntrials-1; % p.featurebin_subsamp: exclude 1 trial from each other bin
        train.trials = 1:Ntrials;
        remove = trial;

        if p.featurebin_subsamp
            if p.iscircular
                edges = 0:(2*pi/p.featurebin_subsamp):(2*pi);
            else
                edges = linspace(min(stimval), max(stimval),p.featurebin_subsamp+1);
            end
            binid = arrayfun(@(x) find(x<=edges, 1), stimval)-1;
            for i = 1:p.featurebin_subsamp
                if binid(trial)==i
                    continue
                else
                bintrials = find(binid==i);
                remove = [remove bintrials(randi(numel(bintrials)))];
                Ntraintrials = Ntraintrials-1;
                end
            end
        end
        train.trials(remove) = [];

        if p.pca
            coeff = pca(squeeze(allsamples(:,:,train.trials))','NumComponents', p.npcs);
            temp_samples = squeeze(allsamples)'*coeff(:,1:p.npcs);
            temp_samples = permute(temp_samples, [3,2,1]);
        else
            temp_samples = allsamples;
        end

        train.resp = resp(train.trials,:,:);
        test.resp = resp(test.trials,:,:);
        train.allsamples = temp_samples(:,:,train.trials);
        test.allsamples = temp_samples(:,:,test.trials);
        train.stimval = stimval(train.trials);
        test.stimval = stimval(test.trials);
        if p.crosstime
            test.stimval = repmat(test.stimval,[NtimePoints, 1]);
        end
        if p.gennull
            train.perm = randperm(numel(train.trials));
            train.stimval = train.stimval(train.perm);
            train.resp = train.resp(train.perm,:,:);
        end
        train.boot.idx = zeros(Ntraintrials, p.nboot);
        train.boot.sets = randi(p.nsets, [1, p.nboot]);

        if p.featurebin_subsamp %ensure cv folds have equally distributed stimvals
            binid = arrayfun(@(x) find(x<=edges, 1), train.stimval)-1;
            trialsperfold = repmat(floor(Ntraintrials/p.cvk), [1,p.cvk]);
            trialsperfold(1:(Ntraintrials-sum(trialsperfold))) = ceil(Ntraintrials/p.cvk);
            vec = train.trials;
            train.cvid = zeros(size(train.trials));
            bintosample = 1;
            for j = 1:numel(trialsperfold)
                tnum = max(floor(trialsperfold(j)/p.featurebin_subsamp), 1);
                try
                    tids = arrayfun(@(x) datasample(vec(binid==x), tnum, 'Replace', false), 1:p.featurebin_subsamp, 'UniformOutput', false);
                catch
                    %                     arrayfun(@(x) disp(vec(binid==x)), 1:p.featurebin_subsamp); disp(tnum);
                    tnum = tnum-1;
                    tids = arrayfun(@(x) datasample(vec(binid==x), tnum, 'Replace', false), 1:p.featurebin_subsamp, 'UniformOutput', false);
                end
                xx = cell2mat(tids);
                binid(ismember(vec,xx)) = []; vec(ismember(vec,xx)) = [];
                while numel(xx)<trialsperfold(j)
                    bintosample = mod(bintosample, p.featurebin_subsamp)+1;
                    if any(binid==bintosample)
                        smpl = datasample(vec(binid==bintosample),1, 'Replace', false);
                        xx = [xx smpl]; binid(vec==smpl) = []; vec(vec==smpl) = [];
                    end
                end
                train.cvid(ismember(train.trials, xx)) = j;
            end
            if p.nboot~=1
                binids = arrayfun(@(x) find(x<=edges, 1), train.stimval)-1;     
                sm = arrayfun(@(x) sum(binids==x), 1:p.featurebin_subsamp);
                cs = [0 cumsum(sm)];
                vec = 1:Ntraintrials;
                for j = 1:p.featurebin_subsamp
                    boots = randsample(vec(binids==j), sm(j)*p.nboot, true);
                    train.boot.idx((cs(j)+1):cs(j+1),:) = reshape(boots, [sm(j), p.nboot]);
                end
            else
                train.boot.idx = [1:Ntraintrials]';
            end
        else
            train.cvid = repmat([1:p.cvk]', ceil(Ntraintrials/p.cvk), 1);
            train.cvid = train.cvid(1:Ntraintrials);
            train.cvid = train.cvid(randperm(size(train.cvid, 1)));

            train.boot.idx = randsample(Ntraintrials, Ntraintrials*p.nboot, true);
            train.boot.idx = reshape(train.boot.idx, [Ntraintrials, p.nboot]);
        end

        if p.covmethod=="signalcov"
            x = reshape(permute(train.allsamples, [1 3 2]), [size(train.allsamples,1)*size(train.allsamples,3), size(train.allsamples,2)]);
            x = x-repmat(mean(x,2),1,size(x,2));
            train.cov = x'*x/size(x,1);
        end
        if p.covmethod=="baselinecov"
            y = train.allsamples(1:floor(p.baselinetimewindow/p.msperbin));
            x = reshape(permute(y, [1 3 2]), [size(y,1)*size(y,3), size(y,2)]);
            x = x-repmat(mean(x,2),1,size(x,2));
            train.cov = x'*x/size(x,1);
        end

        if strcmp(p.regressionmethod, 'FLS')
            fitweights(train.allsamples,train.resp,[], [], p, train.boot); %no output, it saves a persistent ;)
        end
        for m = 1:NtimePoints

            if ~strcmp(p.regressionmethod, 'SVM')
                [outp, uncertainty, posteriors, covp] = TAFKAP_decode(m,train,test,p);

                errors = circ_dist(outp, test.stimval);
                avgerrort = mean(abs(errors))/(2*pi);
                avgunct = mean(uncertainty);

                if p.verbose
                    fprintf('\n t=%s, trial=%d, error=%3.2f, uncertainty=%3.2f. %2.1g percent done', timestr(m), trial, ...
                        avgerrort, avgunct, ((trial-1)*NtimePoints+m)/(NtimePoints*Ntrials));
                end

                if ~p.crosstime
                    errmat(m,trial) = errors;
                    uncmat(m,trial) = uncertainty;
                    postmat(trial,:,m) = posteriors;
                else
                    errmat(m,trial,:) = errors;
                    uncmat(m,trial,:) = uncertainty;
                end
                covparams(m,trial) = covp;

            elseif p.regressionmethod == "SVM"

                [outp, preds] = SVM_decode(m, train, test, p);
                if p.iscircular
                    errors = circ_dist(outp', test.stimval);
                    avgerrort = mean(abs(errors))/(2*pi);
                else
                    errors = test.stimval-outp';
                    avgerrort = mean(abs(errors));
                end

                errmat(m,trial) = errors;

                if p.verbose
                    fprintf('\n t=%s, trial=%d, error=%3.2f', timestr(m), trial, avgerrort);
                end
            end
        end
        fprintf('\n trial %d completed in %d sec', trial, toc(trialtime))
    end

    if ~p.crosstime
        corrmat = diag(corr(abs(errmat)',uncmat'));
    end

end

if strcmp(p.regressionmethod, 'SVM')
    covparams = struct();
end
for i = 1:numel(p.dontsave)
    eval(sprintf('%s = []', p.dontsave{i}));
end
