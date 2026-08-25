function [outp, uncertainty, posteriors, covparams] = TAFKAP_decode(t, train, test, p, ttest,lambda_var, lambda)
% warning('off', 'MATLAB:rankDeficientMatrix');
% warning('off', 'MATLAB:nearlySingularMatrix');
% warning('off', 'MATLAB:singularMatrix');

if nargin<5
    ttest = t;
end
if p.crosstime&&p.loocv
    %treat each timepoint as a separate trial
    ttest = 1:size(test.allsamples, 1);
end
%timepointsxchannelxtrial

train_samples = squeeze(train.allsamples(t,:,:));
test_samples = squeeze(test.allsamples(ttest,:,:));
if p.loocv
    %squeeze(1x64x1) -> 1x64
    %squeeze(1x64xN) -> 64xN
    test_samples = test_samples';
end
train_resp = train.resp;

Ntesttrials = size(test_samples,2);
D = size(train_samples,1);
N = size(train_resp, 1);

if D>=N
    fprintf('\n number of trials smaller than number of dimensions');
    return
end

liks = zeros(Ntesttrials, p.nbinsstimval);

%Fit covariance parameters outside bagging loop (repeat trials would lead to underestimation of trial-to-trial variance)
covparams = fitcov(train, t, p);

b=1;
%bagging loop:
while b<=p.nboot

    set = train.boot.sets(b);
    idx = train.boot.idx(:,b);
    boot_samples = train_samples(:,idx);
    boot_resp = train_resp(idx,:,set);
    boot_cvid = train.cvid(:,idx);
    
    W = fitweights(boot_samples', boot_resp, t, boot_cvid, p, b);

    noise = boot_samples - (boot_resp*W')';

    switch p.covmethod
        case 'TAFKAP'
            covin = W;
        otherwise
            covin = [];
    end

    cov = estcov(noise, p, covparams, covin);

    %cov has to be double precision or else you get bad inverses
    [evec,eval] = eig(cov, 'vector');
    while any(isnan(eval))||any(eval<=0)
        eval(isnan(eval)|eval<=0) = 10^-p.smooth_alpha;
        cov = evec*diag(eval)*evec';
        [evec,eval] = eig(cov, 'vector');
    end

    prec_mat = inv(cov);

    %based on our estimate of W, how would we expect our data to look given
    %each possible stimulus value?
    pred = p.basis_resp(:,:,set)*W';
    %pred = nbins X nelectrodes
    res = bsxfun(@minus, permute(test_samples, [3,1,2]), permute(pred, [1,2,3]) );
    % res = permute(res, [2,1,3]);
    rest = permute(res, [2,1,3]);
    %for each trial, I want nbinsXnelectrodes * prec_mat *
    %nelectrodesXnbins
    tmp = pagemtimes(pagemtimes(res, prec_mat), rest);
    ll = -0.5*squeeze(sum( (repmat(eye(p.nbinsstimval), [1,1,Ntesttrials]) .* tmp), 1));
    if size(ll,1)~=Ntesttrials
        ll = ll';
    end
    probs = exp(ll-max(ll, [], 2));
    probs = bsxfun(@rdivide, probs, sum(probs,2));

    liks = liks+probs;

    if strcmp(p.uncmethod, 'chansum')
        chanresp = inv(W'*W)*W'*test_samples;
        chansum = sum(chanresp, 1);
    end

    b = b+1;

    %   lse = max(ll,[],2) + log(sum(exp(ll-max(ll,[],2)), 2));

end

posteriors = bsxfun(@rdivide, liks, sum(liks,2));
% outp = circ_mean(posteriors,[],2);
pop_vec = posteriors*exp(1i*p.binvals);
[maximum, mxid] = max(posteriors, [], 2);

switch p.outpmethod
    case 'pop_vec'
        outp = angle(pop_vec); %Stimulus estimate (likelihood/posterior means)
    case 'max'
        outp = p.binvals(mxid);
end
switch p.uncmethod
    case 'pop_vec'
        uncertainty = sqrt(-2*log(abs(pop_vec)));
    case 'max'
        uncertainty = maximum.^(-1);
    case 'chansum'
        uncertainty = chansum';
    case 'entropy'
        uncertainty = -sum(posteriors.*log(posteriors),2);
end
if any(isnan(uncertainty))
    fprintf('breaks');
end

end

