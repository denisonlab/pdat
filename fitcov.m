function [covparams] = fitcov(train, t, p)
%inputs: train - need cv splits and stimvals in some cases etc
%t - timepoint
%params - instructions for how to fit or fitted parameters

%covmethods:
%diag_shrink (1 parameter: cv or numeric *lambda [0,1])
%TAFKAP (1 parameter *lambda, lambda_var [0,1])
%FA (1 hyperparameter *k [1, D], remaining parameters fit by EM)
%FA_shrink (shrinks toward FA. 2 hyperparameters, *k [1,D], lambda [0,1]
%generic - given n target components, shrink to target (n parameters
%*lambda, lambda0 ,... [0,1]) not yet implemented, would be nice to explore
%using a mix of pre-stimulus covariance patterns and scov
%none (0 parameters)
%sample (0 parameters)
%independent (0 parameters)
%signal
%baseline

%lossmethods:
%likelihood (negative likelihood of sample covariance of heldout trials given estimate)
%accuracy (mean absolute error when decoding on heldout trials given estimate)
%uncertainty (mean uncertainty when decoding on heldout trials given estimate)
    %the uncertainty option is prone to overfitting (you can come up witn a matrix that gives you
    % epsilon uncertainty no matter what), but may work with suitably constrained designs
%numeric (not cv, there is a formula for optimal lambda in diag_shrink,
%though I don't know how it generalizes when shrink_diag is on)

%options:
%shrink_diag - if method has a diagonal component, also shrink that towards the median (+1 parameter *lambda_var)

warning('off', 'MATLAB:rankDeficientMatrix'); %do i have to d othis every time?
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:illConditionedMatrix');

tofit = {};
nfit = 0;

switch p.covmethod
      case 'diag_shrink'
        if strcmp(p.lossmethod, 'numeric')
            set_lambdas = zeros(1, p.nsets);
            train_samples = squeeze(train.allsamples(t,:,:));
            for set = 1:p.nsets
                bv = 1:p.nboot; wboot = randsample(bv(train.boot.sets==set), 1); %for fls... use a bag iter which uses the same set.       
                W = fitweights(train_samples',train.resp(:,:,set),t,train.cvid,p, wboot);
                noise = train_samples - (train.resp(:,:,set)*W')';
                [d,n] = size(noise);
                scov = noise*noise'/(n-1);
                nu = trace(scov)/d;
                mu = mean(noise,2);
                x = bsxfun(@minus, noise, mu);
                z = zeros(d,d,n);
                for w = 1:n
                    z(:,:,w) = x(:,w)*x(:,w)';
                end
                varz = var(z,0,[1 2]);
                s = scov;
                s(eye(d)==1) = 0;
                set_lambdas(set) = n/((n-1)^2)*sum(varz, 'all')/(sum(s.^2, 'all') + sum((diag(scov)-nu).^2));
            end
            covparams.lambda = mean(set_lambdas);
        else
            nfit = nfit+1;
            tofit{end+1} = 'lambda';
        end


    case 'TAFKAP'
        nfit = nfit+2;
        tofit{end+1} = 'lambda';
        tofit{end+1} = 'lambda_var';

    case 'FA'
        nfit = nfit+1;
        tofit{end+1} = 'k';

    case 'FA_shrink'
        nfit = nfit+2;
        tofit{end+1} = 'k';
        tofit{end+1} = 'lambda';

end
if p.shrink_diag&&(~strcmp(p.covmethod, 'TAFKAP'))
    nfit = nfit+1;
    tofit{end+1} = 'lambda_var';
end

if nfit
    if strcmp(p.lossmethod, 'numeric')
        %fprintf('\n loss method set to numeric but no numeric solution for the provided covariance structure in fitcov.m. using negative log likelihood as loss method.')
        p.lossmethod = 'likelihood';
    end
    %precalculate values for training and testing on each cv fold
    switch p.lossmethod
        case 'likelihood'
            vscov{p.cvk} = [];
        case 'accuracy'
            vsamples{p.cvk} = [];
            vstimval{p.cvk} = [];
            cpred{p.cvk} = [];
        case 'uncertainty'
            vsamples{p.cvk} = [];
            cpred{p.cvk} = [];
    end
    cnoise{p.cvk} = [];
    set = 1;
    tsamp = squeeze(train.allsamples(t,:,:));
    if strcmp(p.covmethod, 'TAFKAP')
        cws{p.cvk} = [];
    end
    for fold = 1:p.cvk
        cids = find(train.cvid~=fold);
        vids = find(train.cvid==fold);
        cresp = train.resp(cids,:,set);
        csamp = tsamp(:,cids);
        bv = 1:p.nboot; wboot = randsample(bv(train.boot.sets==set), 1); %for fls... use a bag iter which uses the same set.
        cw = fitweights(csamp',cresp,t,[],p, wboot);
        if strcmp(p.covmethod, 'TAFKAP')
            cws{fold} = cw;
        end
        cnoise{fold} = csamp - (cresp*cw')';
        %different loss methods require different pre-calculations on
        %validation data
        switch p.lossmethod
            case 'likelihood'
                vresp = train.resp(vids,:,set);
                vsamp = tsamp(:,vids);
                vw = fitweights(vsamp',vresp,t, [], p, wboot);
                vnoise = vsamp - (vresp*vw')';
                vscov{fold} = vnoise*vnoise'/(numel(vids));
            case 'accuracy'
                cpred{fold} = shiftdim((p.basis_resp(:,:,set)*cw')', -1);
                vsamples{fold} = repmat(tsamp(:,vids)', [1, 1, p.nbinsstimval]);
                vstimval{fold} = train.stimval(vids);
            case 'uncertainty'
                cpred{fold} = shiftdim((p.basis_resp(:,:,set)*cw')', -1);
                vsamples{fold} = repmat(tsamp(:,vids)', [1, 1, p.nbinsstimval]);
        end
    end


    %can this one be vectorized?
    ranges{nfit} = [];
    for i = 1:nfit
        ranges{i} = p.([tofit{i}, '_range']);
    end
    s = cellfun(@length, ranges);
    Ngrid = min(max(2, ceil(sqrt(s))), s); %Number of values to visit in each dimension (has to be at least 2, except if there is only 1 value for that dimension)
    grid_vec = cellfun(@(x,y) ceil(linspace(1, y, x)), num2cell(Ngrid), num2cell(s), 'UniformOutput', 0);

    coordinates = cell(size(grid_vec));
    [coordinates{:}] = ndgrid(grid_vec{:});
    searchspace = cell(size(ranges));
    [searchspace{:}] = ndgrid(ranges{:});
    szs = cellfun(@numel, ranges);
    if numel(szs)==1
        sz = [szs 1];
    else
        sz = szs;
    end
    searchn = numel(coordinates{1});
    losses = nan(searchn,1);
    x={};

    for grid_iter=1:searchn
        ids = cellfun(@(x) x(grid_iter), coordinates);
        vals = num2cell(cellfun(@(x,y) x(y), searchspace, num2cell(ids)));
        currparams = cell2struct(vals, tofit, 2);
        loss = 0;
        for fold=1:p.cvk
            switch p.covmethod
                case 'TAFKAP'
                    in = cws{fold};
                case 'FA_shrink' %uses persistents to avoid repeating expectation maximization step
                    in = fold;
                otherwise
                    in = [];
            end
            cov = estcov(cnoise{fold}, p, currparams, in);

            switch p.lossmethod
                case 'likelihood'
                    %                     try
                    %                         lossk = (logdet(cov, 'chol') + sum(sum(invChol_mex(cov).*vscov{fold})))/size(vscov{fold},2);
                    %                     catch ME
                    %                         if any(strcmpi(ME.identifier, {'MATLAB:posdef', 'MATLAB:invChol_mex:dpotrf:notposdef'}))
                    lossk = (log(det(cov)) + trace(cov\vscov{fold}))/size(vscov{fold},2);
                    %                         else
                    %                             rethrow(ME);
                    %                         end
                    %                     end
                    
                case 'accuracy'
                    %look it's vectorized they thought it couldn't be done
                    prec_mat = inv(cov);
                    res = bsxfun(@minus, vsamples{fold}, cpred{fold});
                    res = permute(res, [3,2,1]);
                    rest = permute(res, [2,1,3]);
                    tmp = pagemtimes(pagemtimes(res, prec_mat), rest);
                    ll = -0.5*squeeze(sum(pagemtimes(eye(p.nbinsstimval), tmp), 1));
                    probs = exp(ll-max(ll, [], 1));
                    probs = bsxfun(@rdivide, probs, sum(probs, 1));
                    pop_vec = probs'*exp(1i*p.binvals);
                    outp = angle(pop_vec);
                    lossk = mean(abs(circ_dist(outp, vstimval{fold})));
                case 'uncertainty'
                    prec_mat = inv(cov);
                    res = bsxfun(@minus, vsamples{fold}, cpred{fold});
                    res = permute(res, [3,2,1]);
                    rest = permute(res, [2,1,3]);
                    tmp = pagemtimes(pagemtimes(res, prec_mat), rest);
                    ll = -0.5*squeeze(sum(pagemtimes(eye(p.nbinsstimval), tmp), 1));
                    probs = exp(ll-max(ll, [], 1));
                    probs = bsxfun(@rdivide, probs, sum(probs, 1));
                    pop_vec = probs'*exp(1i*p.binvals);
                    lossk = mean(sqrt(-2*log(abs(pop_vec))));
            end
            if imag(lossk)~=0||isnan(lossk)
                lossk = inf; continue
            end
            loss = loss + lossk;
        end
        losses(grid_iter) = loss;
    end
    [best_loss, best_idx] = min(losses);
    visited = sub2ind(sz,coordinates{:}); visited = visited(:);
    best_idx = visited(best_idx);

    % fprintf('\n--PATTERN SEARCH--');      girl we know
    step_size = 2^floor(log2(diff(coordinates{1}(1:2)/2)));

    stepcell = mat2cell(repmat([-1;1], [1, nfit]), 2, ones(1, nfit));
    steps = cell(size(stepcell));
    [steps{:}] = ndgrid(stepcell{:});
    steps = cellfun(@(x) reshape(x, 1, []), steps, 'UniformOutput', 0);
    steps = cell2mat(steps')';


    while 1
        best_sub = cell(1, nfit);
        [best_sub{:}] = ind2sub(sz, best_idx);
        best_sub = cell2mat(best_sub);
        new_sub = repmat(best_sub, [size(steps, 1), 1]) + steps*step_size;
        del_idx = any(new_sub<=0, 2)|any(new_sub>repmat(szs, [size(steps,1), 1]), 2);
        new_sub(del_idx, :) = [];
        new_sub = mat2cell(new_sub, size(new_sub,1), ones(1,nfit));
        new_idx = sub2ind(sz, new_sub{:});

        new_idx = new_idx(~ismember(new_idx, visited));
        this_losses = nan(size(new_idx));

        if ~isempty(new_idx)
            for ii = 1:length(new_idx)
                vals = num2cell(cellfun(@(x,y) x(y), searchspace, num2cell(repmat(new_idx(ii), 1, nfit))));
                currparams = cell2struct(vals, tofit, 2);
                loss = 0;
                for fold=1:p.cvk
                    switch p.covmethod
                        case 'TAFKAP'
                            in = cws{fold};
                        otherwise
                            in = [];
                    end
                    cov = estcov(cnoise{fold}, p, currparams, in);

                    switch p.lossmethod
                        case 'likelihood'
                            %                     try
                            %                         lossk = (logdet(cov, 'chol') + sum(sum(invChol_mex(cov).*vscov{fold})))/size(vscov{fold},2);
                            %                     catch ME
                            %                         if any(strcmpi(ME.identifier, {'MATLAB:posdef', 'MATLAB:invChol_mex:dpotrf:notposdef'}))
                            lossk = (log(det(cov)) + trace(cov\vscov{fold}))/size(vscov{fold},2);
                            %                         else
                            %                             rethrow(ME);
                            %                         end
                            %                     end
                            
                        case 'accuracy'
                            %vectorize like this in the decoding function!
                            %look it's vectorized they thought it couldn't be done
                            prec_mat = inv(cov);
                            res = bsxfun(@minus, vsamples{fold}, cpred{fold});
                            res = permute(res, [3,2,1]);
                            rest = permute(res, [2,1,3]);
                            tmp = pagemtimes(pagemtimes(res, prec_mat), rest);
                            ll = -0.5*squeeze(sum(pagemtimes(eye(p.nbinsstimval), tmp), 1));
                            probs = exp(ll-max(ll, [], 1));
                            probs = bsxfun(@rdivide, probs, sum(probs, 1));
                            pop_vec = probs'*exp(1i*p.binvals);
                            outp = angle(pop_vec);
                            lossk = mean(abs(circ_dist(outp, vstimval{fold})));
                        case 'uncertainty'
                            prec_mat = inv(cov);
                            res = bsxfun(@minus, vsamples{fold}, cpred{fold});
                            res = permute(res, [3,2,1]);
                            rest = permute(res, [2,1,3]);
                            tmp = pagemtimes(pagemtimes(res, prec_mat), rest);
                            ll = -0.5*squeeze(sum(pagemtimes(eye(p.nbinsstimval), tmp), 1));
                            probs = exp(ll-max(ll, [], 1));
                            probs = bsxfun(@rdivide, probs, sum(probs, 1));
                            pop_vec = probs'*exp(1i*p.binvals);
                            lossk = mean(sqrt(-2*log(abs(pop_vec))));
                    end
                    if imag(lossk)~=0||isnan(lossk)
                        lossk = inf; continue
                    end
                    loss = loss + lossk;
                end
                this_losses(ii) = loss;

            end

            visited = vertcat(visited, new_idx);
            losses = vertcat(losses, this_losses);
       
        
        end
        %fprintf('iter\n')

        if any(this_losses<best_loss)
            [best_loss, best_idx] = min(losses);
            best_idx = visited(best_idx);
        elseif step_size>1
            step_size = step_size/2;
        else
            break
        end
    end

    vals = num2cell(cellfun(@(x,y) x(y), searchspace, num2cell(repmat(best_idx, 1, nfit))));
    covparams = cell2struct(vals, tofit, 2);

%     if p.verbose
%         disp(covparams);
%     end

elseif ~exist('covparams', 'var')
    covparams = struct();
end

end