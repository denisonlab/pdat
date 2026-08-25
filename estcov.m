function cov = estcov(noise, p, params, in)
%inputs: x - noise matrix
%p - settings
%params - fitted parameters
%in - other stuff

%covmethods:
%diag_shrink (1 parameter: cv or numeric *lambda [0,1])
%TAFKAP (2 parameters *lambda, *lambda_var [0,1]) needs in = W
%FA (1 hyperparameter *k [1, D], remaining parameters fit by EM)
%generic - given n target components, shrink to target (n parameters
%*lambda, lambda0 ,... [0,1]) (not implemented yet)
%none (0 parameters)
%sample (0 parameters)
%independent (0 parameters)

%lossmethods:
%likelihood (negative likelihood of sample covariance of heldout trials given estimate)
%accuracy (mean absolute error when decoding on heldout trials given estimate)
%uncertainty (mean uncertainty when decoding on heldout trials given estimate)
%numeric (not cv, there is a formula for optimal lambda in diag_shrink,
%though I don't know how it generalizes when shrink_diag is on)

%options:
%shrink_diag - if method has a diagonal component, also shrink that towards
%its median (+1 parameter *lambda_var)

[d, n] = size(noise);
scov = noise * noise' / (n-1);

switch p.covmethod
    case 'diag_shrink'
        var = scov(eye(d)==1);
        if p.shrink_diag
            var = params.lambda_var * median(var) + (1 - params.lambda_var) * var;
        end
        cov = (1 - params.lambda) * scov;
        cov(eye(d)==1) = cov(eye(d)==1) + params.lambda * var;
    case 'TAFKAP'
        W = in;
        opt = optimoptions('lsqlin');
        opt.Display = ('off');
        dia = tril(ones(d),-1)==1;
        WWt = W*W';
        var = scov(eye(d)==1);
        coeff = lsqlin([WWt(dia), ones(sum(dia(:)),1)], scov(dia), [], [], [], [], [0 0], [], [], opt);
        targetcov = coeff(1)*WWt + coeff(2)*ones(size(W,1));
        targetdiag = params.lambda_var*median(var)+(1-params.lambda_var)*var;
        targetcov(eye(size(W,1))==1) = targetdiag;
        cov = (1-params.lambda)*scov+params.lambda*targetcov;
    case 'FA'
        cov = FA_em(params.k);
        if p.shrink_diag
            var = cov(eye(d)==1);
            var = (1-params.lambda_var)*var + params.lambda_var*median(var);
            cov(eye(d)==1) = var;
        end        %this kind of shrinkage with factor analysis isn't typical so I'm not sure if you would shrink after or before
    case 'FA_shrink'
        if ~isempty(in)
            persistent fa_mat %use a persistent to store intermediate EM results
            if isempty(fa_mat)
                fa_mat = nan(d, d, numel(p.k_range), p.cvk);
            end
            if any(isnan(fa_mat(:,:,params.k,in)))
                fa_cov = FA_em(params.k);
                fa_mat(:,:,params.k, in) = fa_cov;
            else
                fa_cov = squeeze(fa_mat(:,:,params.k,in));
            end
        else
            fa_cov = FA_em(params.k);
        end
        cov = params.lambda*fa_cov + (1-params.lambda)*scov;
        if p.shrink_diag
            var = cov(eye(d)==1);
            var = (1-params.lambda_var)*var + params.lambda_var*median(var);
            cov(eye(d)==1) = var;
        end
    case 'none'
        cov = eye(d);
    case 'sample'
        cov = scov;
        if p.shrink_diag
            var = cov(eye(d)==1);
            var = (1-params.lambda_var)*var + params.lambda_var*median(var);
            cov(eye(d)==1) = var;
        end
    case 'independent'
        cov = scov;
        cov(eye(d)~=1) = 0;
        if p.shrink_diag
            var = cov(eye(d)==1);
            var = (1-params.lambda_var)*var + params.lambda_var*median(var);
            cov(eye(d)==1) = var;
        end
    otherwise
        if isfield(train, 'cov')
            cov = train.cov;
        end
end

if p.smooth_cov
    smoothattempts = 0;
    while smoothattempts<1000 %it really shouldn't take more than one, but it gets crazy with the parameter search
        [evec, eval] = eig(cov);
        eval = real(eval);
        c = 10^(-p.smooth_alpha);
        if ~any(diag(eval)<c, 'all')
            break
        end
        newD = eye(d).*max(eval,c);
        cov = real(evec)*newD*real(evec)';
        smoothattempts = smoothattempts+1;
    end
    if smoothattempts==1000
        cov = eye(d); %use this default so that at least the loss can still be calculated
    end
    %     disp(smoothattempts)
end

    function cov = FA_em(k)
        if k>=d %k must be less than the number of electrodes.
            cov = eye(d);
            return
        end
        %   if ~isfield(params, 'lambda_0') %uses the first k eigenvectors
        [evc, evl] = eigs(scov, k);
        lambda_0 = evc*sqrt(evl);
        %   else
        %      lambda_0 = params.lambda_0;
        %  end
        lambda{1} = lambda_0;
        psi{1} = eye(d).*scov - eye(d).*(lambda_0*lambda_0');
        dev{1} = inf;
        iter = 1;
        xxt =noise * noise';
        while iter<p.FA_iters
            iter = iter+1;
            L = lambda{iter-1};
            P = psi{iter-1};
            beta = L' * inv(L*L' + P);
            Ezzt = n*(eye(k) - beta*L) + beta*xxt*beta';
            newL = (xxt*beta')*inv(Ezzt);
            newP = eye(d).*((xxt-newL*beta*xxt))./n;
            lambda{iter} = newL;
            psi{iter} = newP;
            delta = sum((newL*newL'+newP-L*L'-P).^2, 'all');
            %             disp(delta);
            if delta<p.FA_thresh
                break
            end
        end
        cov = lambda{iter}*lambda{iter}'+psi{iter};
    end
end